package Extractor.Utils;

import java.util.HashMap;
import java.util.Map;

public class TypeSet {
    private static final TypeFactoryImplementation implementation = new TypeFactoryImplementation();

    public static Type type(String value) {
        return implementation.makeOrGetType(value);
    }

    private static class TypeFactoryImplementation extends Type.TypeFactory {
        private final Map<String, Type> types = new HashMap<>();

        public Type makeOrGetType(String name) {
//            return types.computeIfAbsent(name == null ? "" : name, (key) -> makeType(key));
            if (name == null)
                name = "";

            Type sym = types.get(name);
            if (sym == null) {
                sym = makeType(name);
                types.put(name, sym);
            }

            return sym;
        }
    }
}