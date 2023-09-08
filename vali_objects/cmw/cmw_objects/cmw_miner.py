

class CMWMiner:
    def __init__(self, miner_id, wins, o_wins, scores):
        self.miner_id = miner_id
        self.wins = wins
        self.o_wins = o_wins
        self.scores = scores

    def add_score(self, score):
        self.scores.append(score)
