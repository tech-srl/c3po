package Extractor.Model;

import Extractor.Common.Common;

import java.util.List;

public class ASTPathElement {
    private String value;
    private int id;
    private int position;
    private String direction;

    public ASTPathElement(Node node) {
        this.value = splitTokens(node);
        if (isPosZero(node)) {
            this.position = 0;
        } else {
            this.position  = node.getChildIdx();
        }
        this.id= node.getId();
        this.direction = "";
    }

    private static boolean isPosZero(Node node) {
        if (node.getValue().equals(Common.DEL) || node.getValue().equals(Common.UPD) || node.getValue().equals(Common.INS)) {
            return true;
        }
        Node parentNode = node.getParentNode();
        if (parentNode != null && parentNode.getValue().equals(Common.UPD)) {
            return true;
        }
        return false;
    }

    public ASTPathElement(Node node, String direction) {
        this(node);
        this.direction = direction;
    }

    public String getValue() {
        return value;
    }

    public int getId() {
        return id;
    }

    public int getPosition() { return position; }

    @Override
    public String toString() {
        if (direction.equals("")) {
            return value + " " + position;
        }
        return value + " " + position + " " + direction;
    }

    private static String splitTokens(Node node) {
        List<String> splitNameParts = Common.splitToSubtokens(node.getValue());
        String splitName = String.join(Common.internalSeparator, splitNameParts);

        String value = Common.normalizeName(node.getValue(), Common.BlankWord);
        if (value.length() > Common.c_MaxLabelLength) {
            value = value.substring(0, Common.c_MaxLabelLength);
        }
        if (splitName.length() == 0) {
            splitName = value;
        }
        return splitName;
    }
}
