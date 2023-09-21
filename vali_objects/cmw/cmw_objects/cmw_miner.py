

class CMWMiner:
    def __init__(self, miner_id, wins, win_value, scores):
        self.miner_id = miner_id
        self.wins = wins
        self.win_value = win_value
        self.scores = scores

    def add_score(self, score):
        self.scores.append(score)

    def add_win(self):
        self.wins += 1

    def add_win_value(self, add_win_value):
        self.win_value += add_win_value
