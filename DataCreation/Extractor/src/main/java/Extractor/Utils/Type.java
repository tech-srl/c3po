package Extractor.Utils;

import static Extractor.Utils.TypeSet.type;

public final class Type {
    private static int type_counter= 0;

    public final String name;
    public final int type_number;

    public static final Type NO_TYPE = type("");

    private Type(String value) {
        name = value;
        type_number = type_counter;
        type_counter++;
    }

    public boolean isEmpty() {
        return this == NO_TYPE;
    }

    @Override
    public String toString() {
        return name;
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    static class TypeFactory {
        protected TypeFactory() {}

        protected Type makeType(String name) {
            return new Type(name);
        }
    }
}