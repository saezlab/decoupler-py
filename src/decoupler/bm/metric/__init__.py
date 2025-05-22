from decoupler.bm.metric._auc import auc
from decoupler.bm.metric._fscore import fscore
from decoupler.bm.metric._qrank import qrank
from decoupler.bm.metric._hmean import hmean

dict_metric = {
    'auc': auc,
    'fscore': fscore,
    'qrank': qrank,
}
