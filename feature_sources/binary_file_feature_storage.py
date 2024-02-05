# developer: taoshi-mbrown
# Copyright © 2024 Taoshi, LLC
from bz2 import BZ2File
from features import FeatureID, FeatureStorage
import io
import numpy as np
from numpy import ndarray


class BinaryFileFeatureStorage(FeatureStorage):
    SOURCE_NAME = "BinaryFile"

    _ENDIANNESS = "little"

    _MAGIC = b"TaoshiFS"
    _MAGIC_SIZE = len(_MAGIC)
    _VERSION = 1
    _VERSION_SIZE = 1
    _FEATURE_COUNT_SIZE = 2
    _FEATURE_ID_SIZE = 4
    _DTYPE_NUM_SIZE = 1
    _START_TIME_SIZE = 8
    _INTERVAL_SIZE = 8

    def __init__(
        self,
        filename: str,
        mode,
        feature_ids: list[FeatureID],
        feature_dtypes: list[np.dtype] = None,
        default_dtype: np.dtype = np.dtype(np.float32),
        compresslevel: int = 9,
    ):
        # TODO: Allow more modes when compresslevel is 0
        if ("r" in mode) and ("w" in mode):
            raise ValueError("mode cannot include r and w (reading and writing).")
        if "a" in mode:
            raise ValueError("mode cannot include a (appending).")
        if "t" in mode:
            raise ValueError("mode cannot include t (text).")

        self.VALID_FEATURE_IDS = feature_ids
        super().__init__(feature_ids, feature_dtypes, default_dtype)

        self._file_dtypes = []

        row_size = 0
        for feature_dtype in self.feature_dtypes:
            row_size += feature_dtype.itemsize
            self._file_dtypes.append(feature_dtype.newbyteorder(self._ENDIANNESS))
        self._row_size = row_size

        self._start_time_ms = None
        self._sample_count = None
        self._interval_ms = 0
        self._header_size = 0
        self._next_start_time_ms = None

        if compresslevel == 0:
            if "b" not in mode:
                mode += "b"
            self._file = io.open(filename, mode=mode)
        else:
            self._file = BZ2File(
                filename=filename, mode=mode, compresslevel=compresslevel
            )
        self._read_mode = "r" in mode
        if self._read_mode:
            self._read_header()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._file.close()

    @staticmethod
    def _int_from_bytes(buffer: bytes) -> int:
        return int.from_bytes(bytes=buffer, byteorder="little")

    @staticmethod
    def _int_to_bytes(value: int, size: int) -> bytes:
        return value.to_bytes(length=size, byteorder="little")

    def _read_header(self):
        read = self._file.read
        int_from_bytes = self._int_from_bytes

        magic = read(self._MAGIC_SIZE)
        if magic != self._MAGIC:
            raise RuntimeError("Bad header.")

        buffer = read(self._VERSION_SIZE)
        version = int_from_bytes(buffer)
        if version != self._VERSION:
            raise RuntimeError(f"Version {version} not supported.")

        buffer = read(self._FEATURE_COUNT_SIZE)
        feature_count = int_from_bytes(buffer)
        if feature_count != self.feature_count:
            raise RuntimeError(
                f"Feature count {self.feature_count} does not match {feature_count} "
                "value in file."
            )

        for feature_index in range(feature_count):
            buffer = read(self._FEATURE_ID_SIZE)
            header_feature_id = int_from_bytes(buffer)
            buffer = read(self._DTYPE_NUM_SIZE)
            header_dtype_num = int_from_bytes(buffer)

            feature_id = self.feature_ids[feature_index]
            feature_dtype = self.feature_dtypes[feature_index]

            if header_feature_id != feature_id:
                raise RuntimeError(
                    f"Feature index {feature_index} id {header_feature_id} "
                    f"mismatch with {feature_id}."
                )

            if header_dtype_num != feature_dtype.num:
                raise RuntimeError(
                    f"Feature {header_feature_id} dtype {header_dtype_num} "
                    f"mismatch with {feature_dtype.num}."
                )

        buffer = read(self._START_TIME_SIZE)
        self._start_time_ms = int_from_bytes(buffer)

        buffer = read(self._INTERVAL_SIZE)
        self._interval_ms = int_from_bytes(buffer)

        self._header_size = self._file.tell()

    def _write_header(self):
        write = self._file.write
        int_to_bytes = self._int_to_bytes

        write(self._MAGIC)
        write(int_to_bytes(self._VERSION, self._VERSION_SIZE))

        write(int_to_bytes(self.feature_count, self._FEATURE_COUNT_SIZE))
        for feature_index in range(self.feature_count):
            feature_id = self.feature_ids[feature_index]
            feature_dtype = self.feature_dtypes[feature_index]
            buffer = int_to_bytes(feature_id, self._FEATURE_ID_SIZE)
            write(buffer)
            buffer = int_to_bytes(feature_dtype.num, self._DTYPE_NUM_SIZE)
            write(buffer)

        write(int_to_bytes(self._start_time_ms, self._START_TIME_SIZE))
        write(int_to_bytes(self._interval_ms, self._INTERVAL_SIZE))

        self._header_size = self._file.tell()

    def get_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        sample_count: int,
    ) -> dict[FeatureID, ndarray]:
        if not self._read_mode:
            raise RuntimeError("Reading not supported in write mode.")

        if interval_ms != self._interval_ms:
            raise RuntimeError(
                f"Interval mismatch between file {self._interval_ms} "
                f"and request {interval_ms}."
            )

        feature_count = self.feature_count
        file_dtypes = self._file_dtypes
        read = self._file.read

        sample_location = (
            self._header_size
            + int((start_time_ms - self._start_time_ms) / self._interval_ms)
            * self._row_size
        )
        self._file.seek(sample_location)

        feature_samples = []
        for feature_index in range(self.feature_count):
            file_dtype = file_dtypes[feature_index]
            feature_samples.append(np.empty(sample_count, file_dtype))

        for sample_index in range(sample_count):
            for feature_index in range(feature_count):
                file_dtype = file_dtypes[feature_index]
                itemsize = file_dtype.itemsize
                buffer = read(itemsize)
                if len(buffer) != itemsize:
                    raise RuntimeError("Unexpected end of file.")
                sample = np.frombuffer(buffer=buffer, dtype=file_dtype, count=1)
                feature_samples[feature_index][sample_index] = sample

        results = {}
        for feature_index, feature_id in enumerate(self.feature_ids):
            feature_dtype = self.feature_dtypes[feature_index]
            results[feature_id] = feature_samples[feature_index].astype(
                feature_dtype, copy=False
            )

        return results

    def set_feature_samples(
        self,
        start_time_ms: int,
        interval_ms: int,
        feature_samples: dict[FeatureID, ndarray],
    ):
        if self._read_mode:
            raise RuntimeError("Writing not supported in read mode.")

        feature_count = self.feature_count
        write = self._file.write

        if self._start_time_ms is None:
            self._start_time_ms = start_time_ms
            self._sample_count = 0
            self._interval_ms = interval_ms
            self._next_start_time_ms = start_time_ms
            self._write_header()

        if start_time_ms != self._next_start_time_ms:
            raise RuntimeError(
                f"Expected start time of {self._next_start_time_ms} "
                f"but {start_time_ms} requested."
            )

        sample_count = None
        write_samples = []
        for feature_index, feature_id in enumerate(self.feature_ids):
            feature_sample = feature_samples.get(feature_id, None)
            if feature_sample is None:
                RuntimeError(f"Feature {feature_id} missing from feature_samples.")
            feature_sample_count = len(feature_sample)
            if sample_count is None:
                sample_count = feature_sample_count
            elif feature_sample_count != sample_count:
                raise RuntimeError(
                    f"Feature {feature_id} has {feature_sample_count}"
                    f" samples when {sample_count} expected."
                )
            file_dtype = self.feature_dtypes[feature_index]
            feature_sample = feature_sample.astype(file_dtype)
            write_samples.append(feature_sample)

        for sample_index in range(sample_count):
            for feature_index in range(feature_count):
                write(write_samples[feature_index][sample_index].tobytes())

        self._sample_count += sample_count
        self._next_start_time_ms += interval_ms * sample_count

    def get_start_time_ms(self) -> int | None:
        return self._start_time_ms

    def get_sample_count(self) -> int:
        if self._sample_count is None:
            self._file.seek(0, io.SEEK_END)
            file_size = self._file.tell()
            self._sample_count = int((file_size - self._header_size) / self._row_size)
        return self._sample_count
