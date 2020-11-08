package Extractor.Model;

public abstract class MyAction {

    protected int srcId;
    protected int tgtId;
    protected String name;
    protected int subTreeSize;

    public MyAction(int srcId, int tgtId, String name, int subTreeSize) {
        this.srcId = srcId;
        this.tgtId = tgtId;
        this.name = name;
        this.subTreeSize = subTreeSize;
    }

    public int getSrcId() {
        return srcId;
    }

    public int getTgtId() {
        return tgtId;
    }

    public void setSrcId(int srcId) {
        this.srcId = srcId;
    }

    public void setTgtId(int tgtId) {
        this.tgtId = tgtId;
    }

    public int getSubTreeSize() {
        return this.subTreeSize;
    }

    public String getName() {
        return this.name;
    }

    @Override
    public String toString() {
        return getName() + " " + srcId + " to " + tgtId;
    }



}