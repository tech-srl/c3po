package Extractor.Model;

import Extractor.Common.Common;

public class MyUpdate extends MyAction{
    public MyUpdate(int srcId, int tgtId) {
        super(srcId, tgtId, Common.UPD, 1);
    }

}
