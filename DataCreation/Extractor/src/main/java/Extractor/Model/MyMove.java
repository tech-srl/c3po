package Extractor.Model;

import Extractor.Common.Common;

public class MyMove extends MyAction {
    public MyMove(int srcId, int tgtId, int subTreeSize) {
        super(srcId, tgtId, Common.MOV, subTreeSize);
    }
}
