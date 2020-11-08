package Extractor.Model;

public class MutableInteger {
    private int value;

    public MutableInteger(int value) {
        this.value = value;
    }

    public int value() {
        return this.value;
    }

    public int getAndInc() {
        int old = this.value;
        this.value++;
        return old;
    }

    public int getAndDec() {
        int old = this.value;
        this.value--;
        return old;
    }


}
