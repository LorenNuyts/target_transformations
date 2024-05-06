from dataclasses import dataclass
from typing import Optional

import numpy as np

# from experiments.utils import isworse


@dataclass(init=False)
class AlphaRecord:
    lo: float
    hi: float
    alpha: float

    mtrain_clf: float
    mvalid_clf: float
    fit_time: float

    intercept: Optional[np.ndarray]
    coefs: Optional[np.ndarray]

    def __init__(self, lo, hi, alpha):
        self.lo = lo
        self.hi = hi
        self.alpha = alpha
        self.mtrain_clf = 0.0
        self.mvalid_clf = 0.0
        self.fit_time = 0.0
        self.intercept = None
        self.coefs = None


class AlphaSearch:
    def __init__(self):
        self.round = 0
        self.step = 1
        self.lo = -3
        self.hi = 4

        self.records = []
        self.alpha_search_round_steps = [16, 8, 4]

        self.set_lohis()

    def __iter__(self):
        return self

    def __next__(self):
        nsteps = self.nsteps()
        if self.step > nsteps:  # next round
            lo, hi = self.next_lohis()

            self.lo, self.hi = lo, hi
            self.set_lohis()
            self.step = 1

        lo, mid, hi = self.lohis[self.step - 1 : self.step + 2]
        alpha = np.power(10.0, mid)

        record = AlphaRecord(lo, hi, alpha)
        self.records.append(record)

        self.step += 1
        return record

    # def isworse_tr(self, mtrain):
    #     return isworse(mtrain, self.compress.mtrain)
    #
    # def isworse_va(self, mvalid):
    #     return isworse(mvalid, self.compress.mvalid)
    #
    # def isnotworse_tr(self, mtrain):
    #     return not isworse(mtrain, self.compress.mtrain)
    #
    # def isnotworse_va(self, mvalid):
    #     return not isworse(mvalid, self.compress.mvalid)

    # def overfits(self, mtrain, mvalid):
    #     cond1 = self.isnotworse_tr(mtrain)
    #     cond2 = self.isworse_va(mvalid)
    #     cond3 = isworse(mvalid, mtrain)
    #     return cond1 and cond2 and cond3
    #
    # def underfits(self, mtrain, mvalid):
    #     cond1 = self.isworse_tr(mtrain)
    #     cond2 = self.isworse_va(mvalid)
    #     return cond1 and cond2

    def nsteps(self):
        return self.alpha_search_round_steps[self.round]

    def set_lohis(self):
        nsteps = self.nsteps()
        self.lohis = np.linspace(self.lo, self.hi, nsteps + 2)

    # def quality_filter(self, records):
    #     filt = filter(
    #         lambda r:
    #         # self.isnotworse_va(r.mvalid_clf) and
    #         # r.frac_removed < 1.0 and
    #         not isworse(r.mtrain_clf, r.mvalid_clf),  # overfitting
    #         records,
    #     )
    #     return filt

    def next_lohis(self):
        nsteps = self.nsteps()
        num_rounds = len(self.alpha_search_round_steps)

        self.round += 1
        if self.round >= num_rounds:
            raise StopIteration()

        prev_round_records = self.records[-1 * nsteps:]

        # # Filter out the records that are not good enough ...
        # filt = self.quality_filter(prev_round_records)
        # ... and pick the one with the lowest error
        best = min(prev_round_records, default=None, key=lambda r: r.mvalid_clf)

        # # If nothing was good enough, find the last record that was overfitting and the
        # # first that was underfitting, and look at that transition in more detail
        # r_over, r_under = None, None
        # if best is None:
        #     for r in prev_round_records:
        #         if self.overfits(r.mtrain_clf, r.mvalid_clf):
        #             r_over = r
        #         elif r_over is not None and self.underfits(r.mtrain_clf, r.mvalid_clf):
        #             r_under = r
        #             break
        #     if r_over is not None and r_under is not None:
        #         lo = (r_over.lo + r_over.hi) / 2.0
        #         hi = (r_under.lo + r_under.hi) / 2.0
        #         return lo, hi
        #     else:
        #         raise StopIteration()
        # else:
        return best.lo, best.hi

    def get_best_record(self):
        # filt = self.quality_filter(self.records)
        return min(self.records, default=None, key=lambda r: r.mvalid_clf)
