package Extractor.Model;

import Extractor.Common.Common;

public class MyDelete extends MyAction {
    public MyDelete(int srcId, int tgtId, int subTreeSize) {
        super(srcId, tgtId, Common.DEL, subTreeSize);
    }

}

