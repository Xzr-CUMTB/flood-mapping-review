"""Development-only weighted operating point selection with explicit handling of ties."""
import numpy as np


def match_positive_rate(probability, weight, target_rate):
    p,w=np.asarray(probability,dtype=np.float64),np.asarray(weight,dtype=np.float64)
    if len(p)!=len(w) or not len(p) or not (0<=target_rate<=1):raise ValueError('Invalid operating point input')
    if not np.isfinite(p).all() or not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Invalid probabilities or weights')
    order=np.argsort(-p,kind='stable');p,w=p[order],w[order]
    cumulative=np.cumsum(w);ends=np.r_[np.flatnonzero(p[1:]!=p[:-1]),len(p)-1]
    rates=np.r_[0.,cumulative[ends]/cumulative[-1]]
    thresholds=np.r_[np.nextafter(p[0],np.inf),p[ends]]
    index=int(np.argmin(np.abs(rates-target_rate)))
    return {'threshold':float(thresholds[index]),'achieved_rate':float(rates[index]),
            'target_rate':float(target_rate),'absolute_rate_gap':float(abs(rates[index]-target_rate)),
            'rule':'Closest attainable area-weighted rate with probability >= threshold; ties kept together; lower rate wins equal-distance ties'}
