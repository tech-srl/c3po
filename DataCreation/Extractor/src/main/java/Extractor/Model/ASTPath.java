package Extractor.Model;

import Extractor.Common.Common;
import org.javatuples.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class ASTPath {
    private static String UP_SYMBOL = "<U>";
    private static String DOWN_SYMBOL = "<D>";
    private static String SEP = " ";
    private List<ASTPathElement> path;
    private int sourceId;
    private int targetId;
    private String op;

    public ASTPath(int sourceId, int targetId){
        path = new ArrayList<>();
        this.sourceId = sourceId;
        this.targetId = targetId;
        this.op = Common.NOP;
    }

    public int getSize() { return path.size(); }

    public void addNode(Node node) {
        ASTPathElement pathElement = new ASTPathElement(node);
        this.path.add(pathElement);
    }

    public void addNodeUp(Node node) {
        ASTPathElement pathElement = new ASTPathElement(node, UP_SYMBOL);
        this.path.add(pathElement);
    }
    public void addNodeDown(Node node) {
        ASTPathElement pathElement = new ASTPathElement(node, DOWN_SYMBOL);
        this.path.add(pathElement);
    }

    public boolean isEmpty() {
        return path.isEmpty();
    }

    public void setIds(int sourceId, int targetId) {
        this.sourceId = sourceId;
        this.targetId = targetId;
    }

    public void setOp(String op) {
        this.op = op;
    }

    public int getSourceId() {
        return sourceId;
    }

    public int getTargetId() {
        return targetId;
    }

    public String getOp() {
        return op;
    }

    public Map<Pair<Integer,Integer>,ASTPath> getAllSubPathsMap() {
        Map<Pair<Integer,Integer>,ASTPath> map = new HashMap<>();
        for (int i = 0; i < path.size(); i++) {
            for (int j = 0; j < i; j++) {
                List<ASTPathElement> pathList = path.subList(j, i+1);
                int srcId = path.get(j).getId();
                int tgtId = path.get(i).getId();
                ASTPath astPath = new ASTPath(srcId, tgtId);
                astPath.path = pathList;
                map.put(new Pair<>(srcId, tgtId), astPath);
            }
        }
        return map;
    }

    @Override
    public String toString() {
        return path.stream().map(ASTPathElement::toString).collect(Collectors.joining(SEP));
    }

    public String FormattedOpString() {
        String lastElement = path.get(path.size()-1).getValue();
        if(lastElement.equalsIgnoreCase(Common.DEL)) {
            return Common.DEL + " " + sourceId;
        }
        return Common.MOV + " " + sourceId + " " + targetId;
    }

    public String getDotStringFormat(int id) {
        StringJoiner nodesJoiner = new StringJoiner(" ");
        StringJoiner edgesJoiner = new StringJoiner(" -> ");
        for (ASTPathElement e : path) {
            String value = e.getValue().replaceAll("<", "\\\\<").replaceAll(">", "\\\\>").replaceAll("\\|", "\\\\|");
            nodesJoiner.add(id + " [label=\"{"  + value + "}\"];");
            edgesJoiner.add("" + id);
            id++;
        }
        return nodesJoiner.toString() + edgesJoiner + ";";
    }



}
