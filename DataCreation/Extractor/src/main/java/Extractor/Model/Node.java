package Extractor.Model;

import Extractor.Common.Common;
import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import com.github.gumtreediff.tree.TreeUtils;
import org.javatuples.Pair;
import org.javatuples.Triplet;

import java.util.*;
import java.util.stream.Collectors;


public class Node {
    private int id;
    private String value;
    private String type;
    private transient String action;
    private int childIdx;
    private int OpchildIdx;
    private transient int updPointer;
    private transient boolean delPointer;
    private transient int movPointerParent;
    private transient int movSiblingPosition;
    private transient List<Integer> cpyPointerParent;
    private transient List<Integer> cpySiblingPosition;
    private List<Node> children;
    private transient Node parent;
    private transient ITree originalNode;

    private static String NOP = Common.NOP;
    private static String DEL = Common.DEL;
    private static String UPD = Common.UPD;
    private static String MOV = Common.MOV;
    private static String INS = Common.INS;
    public static String SEP = " ";
    private final static String UP_SYMBOL = "<U>";
    private final static String DOWN_SYMBOL = "<D>";
    private final static String DUMMY = "DUMMY";
    public final static String NUM_LITERAL = "NUM";
    public final static String STRING_LITERAL = "STR";
    public final static String CHAR_LITERAL = "CHAR";
    public final static String LITERAL = "LITERAL";
    public static Map<String,Set<String>> legalChildMap = null;

    private Map<Pair<Integer, Integer>, ASTPath> pathsMap = new HashMap<>();


    public Node(ITree originalNode, int id, String type, String value, Node parent, String action, List<Integer> updPointer,
                int movPointerParent, int movSiblingPosition, boolean delPointer,
                List<Integer> cpyPointerParent, List<Integer> cpySiblingPosition) {
        this.originalNode = originalNode;
        this.id = id;
        this.value = value;
        this.children = new ArrayList<>();
        this.action = action;
        if (updPointer.size() == 0) {
            this.updPointer = -1;
        } else {
            this.updPointer = Collections.min(updPointer);
        }
        this.movPointerParent = movPointerParent;
        this.movSiblingPosition = movSiblingPosition;
        this.delPointer = delPointer;
        this.cpyPointerParent = cpyPointerParent;
        this.cpySiblingPosition = cpySiblingPosition;
        this.parent = parent;
        this.OpchildIdx = -1;
        this.type = type;
    }

    public Node(int id, String type, String value, Node parent) {
        this.id = id;
        this.value = value;
        this.type = type;
        this.children = new ArrayList<>();
        this.parent = parent;
        this.action = NOP;
        this.updPointer = -1;
        this.movPointerParent = -1;
        this.movSiblingPosition = -1;
        this.delPointer = false;
        this.OpchildIdx = -1;
        this.cpyPointerParent = new ArrayList<>();
        this.cpySiblingPosition = new ArrayList<>();
    }

    public Node(Node node) {
        this.id = node.id;
        this.value = node.value;
        this.type = node.type;
        this.children = new ArrayList<>();
        this.parent = null;
        this.action = node.action;
        this.updPointer = node.updPointer;
        this.movPointerParent = node.movPointerParent;
        this.movSiblingPosition = node.movSiblingPosition;
        this.delPointer = node.delPointer;
        this.OpchildIdx = node.OpchildIdx;
        this.cpyPointerParent = new ArrayList<>(node.cpyPointerParent);
        this.cpySiblingPosition = new ArrayList<>(node.cpySiblingPosition);
        this.originalNode = node.originalNode;
    }

    public void addChild(Node child) {
        this.children.add(child);
        child.setChildIdx(this.children.size() - 1);
        child.parent = this;
    }

    public Node setChildIdx(int idx) {
        this.childIdx = idx;
        return this;
    }

    public int getId() {
        return this.id;
    }

    public int getChildIdx() {
        return this.childIdx;
    }

    public String getValue() {
        return this.value;
    }

    public Node getParentNode() {
        return this.parent;
    }

    public List<Node> getChildren() {
        return this.children;
    }

    public boolean isLeaf() {
        return this.children.size() == 0;
    }

    private static Node getNodeById(Node node, int id) {
        if (node.id == id) {
            return node;
        }
        if (node.children != null) {
            for (Node child : node.children) {
                Node res = getNodeById(child, id);
                if (res != null) {
                    return res;
                }
            }
        }
        return null;
    }

    public static Node clone(Node parent, Node tree) {
        Node node = new Node(tree);
        node.parent = parent;
        node.setChildIdx(0);
        if (!tree.isLeaf()) {
            int i = 1;
            for (Node child : tree.getChildren()) {
                Node nChild = clone(node, child);
                if (nChild != null) {
                    node.addChild(nChild);
                    i++;
                }
            }
        }
        return node;
    }

    private static void applyAction(Action action, Node nodeOfInterest, Node parentOfInterest) {
        if (action instanceof Delete) {
            nodeOfInterest.parent.children.remove(nodeOfInterest);
        } else if (action instanceof Move) {
            int pos = ((Move) action).getPosition() + 1;
            parentOfInterest.children.add(pos, nodeOfInterest);
            nodeOfInterest.parent.children.remove(nodeOfInterest);
            nodeOfInterest.parent = parentOfInterest;
        } else if (action instanceof Insert) {
            Node clonedNode = clone(null, nodeOfInterest);
            parentOfInterest.children.add(((Insert) action).getPosition() + 1, clonedNode);
            clonedNode.parent = parentOfInterest;
        } else if (action instanceof Update) {
            // nothing to do
        }
    }

    private static List<MyAction> createMyActionsNoContext(Node node, ITree beforeTree, List<Action> actions) {
        Set<ITree> iTreeNodes = new HashSet<>(TreeUtils.breadthFirst(beforeTree));
        Node clonedNode = clone(null, node);
        List<MyAction> myActions = new ArrayList<>();
        List<Node> nodeList = getNodesList(clonedNode);
        // we cant refer to inserted nodes since they are not exists in the original AST
        Set<Node> inserted = new HashSet<>();
        int delNodeId = nodeList.stream().filter(n -> n.value.equals(DEL)).collect(Collectors.toList()).get(0).id;
        for (Action action : actions) {
            List<Node> nl = nodeList.stream().filter(n -> Objects.equals(n.id, action.getNode().getId())).collect(Collectors.toList());
            if (nl.size() == 0) { // for example DEL of comment..
                continue;
            }
            Node nodeOfInterest = null;
            nodeOfInterest = nl.get(0);
            boolean isComment = false;
            if (nodeOfInterest.value.startsWith(Common.COMMENT_PREFIX)) {
                isComment = true;
            }
            if (inserted.contains(nodeOfInterest)) {
                return null;
            }
            if (action instanceof Delete) {
                if (!isComment) {
                    myActions.add(new MyDelete(nodeOfInterest.id, delNodeId, Node.getSizeNoDummies(nodeOfInterest)));
                }
                applyAction(action, nodeOfInterest, null);
            } else if (action instanceof Move) {
                ITree parentITreeNode = ((Move) action).getParent();
                if (!iTreeNodes.contains(parentITreeNode)) {
                    return null;
                }
                Node parentNodeOfInterest = nodeList.stream().filter(n -> Objects.equals(n.id, parentITreeNode.getId())).collect(Collectors.toList()).get(0);
                int pos = ((Move) action).getPosition();
                Node sibling = parentNodeOfInterest.children.get(pos);
                Node defactoSibling = sibling;
                if (defactoSibling.value.startsWith(Common.COMMENT_PREFIX)) {
                    for (int i = pos - 1; 0 <= i; i--) {

                        if (!parentNodeOfInterest.children.get(i).value.startsWith(Common.COMMENT_PREFIX)) {
                            defactoSibling = parentNodeOfInterest.children.get(i);
                        }
                    }
                }
                if (inserted.contains(defactoSibling)) {
                    return null;
                }
                if (!isComment) {
                    myActions.add(new MyMove(nodeOfInterest.id, defactoSibling.id, Node.getSizeNoDummies(nodeOfInterest)));
                }
                applyAction(action, nodeOfInterest, parentNodeOfInterest);
            } else if (action instanceof Insert) {
                ITree parentITreeNode = ((Insert) action).getParent();
                if (!iTreeNodes.contains(parentITreeNode)) {
                    return null;
                }
                Node parentNodeOfInterest = nodeList.stream().filter(n -> Objects.equals(n.id, parentITreeNode.getId())).collect(Collectors.toList()).get(0);
                int pos = ((Insert) action).getPosition();
                Node sibling = parentNodeOfInterest.children.get(pos);
                Node defactoSibling = sibling;
                if (defactoSibling.value.startsWith(Common.COMMENT_PREFIX)) {
                    for (int i = pos - 1; 0 <= i; i--) {

                        if (!parentNodeOfInterest.children.get(i).value.startsWith(Common.COMMENT_PREFIX)) {
                            defactoSibling = parentNodeOfInterest.children.get(i);
                        }
                    }
                }
                if (inserted.contains(defactoSibling)) {
                    return null;
                }
                if (!isComment) {
                    myActions.add(new MyInsert(nodeOfInterest.id, defactoSibling.id, Node.getSizeNoDummies(nodeOfInterest)));
                }
                applyAction(action, nodeOfInterest, parentNodeOfInterest);
                inserted.addAll(getNodesList(parentNodeOfInterest.children.get(pos + 1)));
            } else if (action instanceof Update) {
                if (!isComment) {
                    myActions.add(new MyUpdate(nodeOfInterest.id, nodeOfInterest.updPointer));
                }
            }
        }
        return myActions;
    }

    public static List<MyAction> createMyActions(Node node, ITree beforeTree, List<Action> actions, boolean isContext, TreeContext ctx, MutableInteger nextId) {
        if (isContext) {
            return createMyActionsContext(node, beforeTree, actions, ctx, nextId);
        } else {
            return createMyActionsNoContext(node, beforeTree, actions);
        }
    }

    private static List<MyAction> createMyActionsContext(Node node, ITree beforeTree, List<Action> actions, TreeContext ctx, MutableInteger nextId) {
        Set<ITree> iTreeNodes = new HashSet<>(TreeUtils.breadthFirst(beforeTree));
        Node clonedNode = clone(null, node);
        List<MyAction> myActions = new ArrayList<>();
        List<Node> nodeList = getNodesList(clonedNode);
        List<Node> originalNodeList = getNodesList(node);
        Set<ITree> delNodes = actions.stream().filter(x -> x instanceof Delete)
                .map(Action::getNode)
                .map(TreeUtils::breadthFirst)
                .flatMap(List::stream)
                .collect(Collectors.toSet());

        // we cant refer to inserted nodes since they are not exists in the original AST
//        Set<Node> inserted = new HashSet<>();
        int delNodeId = nodeList.stream().filter(n -> n.value.equals(DEL)).collect(Collectors.toList()).get(0).id;
        int insNodeId = originalNodeList.stream().filter(n -> n.value.equals(INS)).collect(Collectors.toList()).get(0).id;
        Node updNode = originalNodeList.stream().filter(n -> n.value.equals(UPD)).collect(Collectors.toList()).get(0);
        for (Action action : actions) {
            List<Node> nl = nodeList.stream().filter(n -> Objects.equals(n.id, action.getNode().getId())).collect(Collectors.toList());
            if (nl.size() == 0) { // for example DEL of comment..
                continue;
            }
            Node nodeOfInterest = null;
            boolean isComment = false;
            try {
                nodeOfInterest = nl.get(0);
                if (nodeOfInterest.value.startsWith(Common.COMMENT_PREFIX)) {
                    isComment = true;
                }
            } catch (IndexOutOfBoundsException e) {
                // pass
            }
            if (action instanceof Delete) {
                if (!isComment) {
                    myActions.add(new MyDelete(nodeOfInterest.id, delNodeId, Node.getSizeNoDummies(nodeOfInterest)));
                }
                applyAction(action, nodeOfInterest, null);
            } else if (action instanceof Move) {
                ITree parentITreeNode = ((Move) action).getParent();
                if (!iTreeNodes.contains(parentITreeNode)) {
                    if (!delNodes.contains(action.getNode())) {
                        if (!isComment) {
                            myActions.add(new MyDelete(nodeOfInterest.id, delNodeId, Node.getSizeNoDummies(nodeOfInterest)));
                        }
                        applyAction(new Delete(null), nodeOfInterest, null);
                    }
                    continue;
                }
                Node parentNodeOfInterest = nodeList.stream().filter(n -> Objects.equals(n.id, parentITreeNode.getId())).collect(Collectors.toList()).get(0);
                int pos = ((Move) action).getPosition();
                Node sibling = parentNodeOfInterest.children.get(pos);
                Node defactoSibling = sibling;
                if (defactoSibling.value.startsWith(Common.COMMENT_PREFIX)) {
                    for (int i = pos - 1; 0 <= i; i--) {
                        if (!parentNodeOfInterest.children.get(i).value.startsWith(Common.COMMENT_PREFIX)) {
                            defactoSibling = parentNodeOfInterest.children.get(i);
                        }
                    }
                }
//                if (inserted.contains(defactoSibling)) {
//                    return null;
//                }
                if (!isComment) {
                    myActions.add(new MyMove(nodeOfInterest.id, defactoSibling.id, Node.getSizeNoDummies(nodeOfInterest)));
                }
                applyAction(action, nodeOfInterest, parentNodeOfInterest);
            } else if (action instanceof Insert) {
                ITree parentITreeNode = ((Insert) action).getParent();
                if (!iTreeNodes.contains(parentITreeNode)) {
                    return null;
                }
                Node parentNodeOfInterest = originalNodeList.stream().filter(n -> Objects.equals(n.id, parentITreeNode.getId())).collect(Collectors.toList()).get(0);
                int pos = ((Insert) action).getPosition() + 1;
                ; // +1 because of the DUMMY node
                if (parentNodeOfInterest.children.size() == 0) {
                    parentNodeOfInterest.addChild(new Node(nextId.getAndInc(), DUMMY, DUMMY, node), 0);
                }

                nodeOfInterest = parentNodeOfInterest.addChildFromITree(pos, action.getNode(), ctx, nextId);
                if (!nodeOfInterest.value.startsWith(Common.COMMENT_PREFIX)) {
                    myActions.add(new MyInsert(nodeOfInterest.id, insNodeId, Node.getSizeNoDummies(nodeOfInterest)));
                }
//                inserted.add(nodeOfInterest);
            } else if (action instanceof Update) {
                if (!isComment) {
                    String value = ((Update) action).getValue();
                    Node toUpd = null;
                    for (Node child : updNode.children) {
                        if (child.getValue().equals(value)) {
                            toUpd = child;
                            break;
                        }
                    }
                    if (toUpd == null) {
                        toUpd = new Node(nextId.getAndInc(), "", value, updNode);
                        updNode.addChild(toUpd);
                    }
                    myActions.add(new MyUpdate(nodeOfInterest.id, toUpd.id));
                }
            }
        }
        return myActions;
    }

    public static String fixLiteral(String literal) {
        if (literal.equals("null")) {
            return literal;
        }
        if (literal.startsWith("\'") && literal.endsWith("\'")) {
            return CHAR_LITERAL;
        }
        if ((literal.startsWith("\"") || literal.startsWith("@\"")) && literal.endsWith("\"")) {
            return STRING_LITERAL;
        }
        if (literal.equals("true") || literal.equals("false")) {
            return literal;
        }
        if (literal.startsWith(".")) {
            return NUM_LITERAL;
        }
        if (literal.toLowerCase().endsWith("u") || literal.toLowerCase().endsWith("f") ||
                literal.toLowerCase().endsWith("l") || literal.toLowerCase().endsWith("d") ||
                literal.toLowerCase().endsWith("e") || literal.toLowerCase().endsWith("m") ||
                literal.toLowerCase().endsWith("s")) {
            literal = literal.substring(0, literal.length() - 1);
        }
        int base = 10;
        if (literal.startsWith("0x") || literal.startsWith("0X")) {
            literal = literal.substring(2);
            base = 16;
        }
        if (literal.startsWith("0b") || literal.startsWith("0B") || literal.contains("_")) {
            literal = literal.substring(2);
            literal = literal.replaceAll("_", "");
            base = 2;
        }
        if (literal.equals("")) {
            return LITERAL;
        }
        Integer value = null;
        try {
            value = Integer.parseInt(literal, base);
        } catch (Exception e) {
            try {
                value = Integer.parseInt(literal, 16);
            } catch (Exception ee) {
//                System.out.println(ee);
            }
        }
        if (value != null) {
            if (value != 0 && value != 1) {
                return NUM_LITERAL;
            }
            return literal;
        }
        return LITERAL;
    }

    public static void clearCommentNodes(Node node) {
        List<Node> newChildren = new ArrayList<>();
        int i = 0;
        for (Node child : node.children) {
            if (!child.getValue().startsWith(Common.COMMENT_PREFIX)) {
                newChildren.add(child);
                child.setChildIdx(i);
                i++;
                clearCommentNodes(child);
            }
        }
        node.children = newChildren;
    }

    private Node addChildFromITree(int pos, ITree tree, TreeContext ctx, MutableInteger nextId) {
        String type = ctx.getTypeLabel(tree.getType());
        String label = type;
        if (!tree.getLabel().equals("")) {
            if (label.equals("literal")) {
                label = fixLiteral(tree.getLabel());
            } else if (label.equals("name")) {
                label = tree.getLabel();
            } else {
                label += "_" + tree.getLabel();
            }
        } else if (tree.getChildren().size() == 0 && label.equals("argument")) {
            // A tree contains node "argument" that is not a leaf  and without children is considered to be bad parsed
            throw new IllegalStateException();
        }
        Node child = new Node(nextId.getAndInc(), type, label, this);
        this.addChild(child, pos);
        return child;
    }

    private void addChild(Node child, int pos) {
        children.add(pos, child);
        for (int i = 0; i < children.size(); i++) {
            children.get(i).setChildIdx(i);
        }
    }


    public static Node createTreePreOrder(Node parent, ITree tree, TreeContext ctx, Map<Integer, ITree> delMap,
                                          Map<Integer, Pair<ITree, List<Integer>>> updatesMap,
                                          Map<Integer, Triplet<ITree, ITree, Integer>> movMap,
                                          Map<Integer, Pair<ITree, List<Insert>>> cpyMap, MutableInteger nextId,
                                          boolean isContext, boolean recordChildren) throws IllegalStateException {
        String type = ctx.getTypeLabel(tree.getType());
        String label = type;
//        if (label.equals("comment")) {
//            return null;
//        }


        boolean isLeaf = false;
        if (!tree.getLabel().equals("")) {
            isLeaf = true;
            if (label.equals("literal")) {
                label = fixLiteral(tree.getLabel());
            } else if (label.equals("name")) {
                label = tree.getLabel();
            } else {
                label += "_" + tree.getLabel();
            }
        } else if (tree.getChildren().size() == 0 && label.equals("argument") && !isContext) {
            // A tree contains node "argument" that is not a leaf  and without children is considered to be bad parsed
            throw new IllegalStateException();
        }
        String action = NOP;
        List<Integer> updPointer = new ArrayList<>();
        int movPointerParent = -1;
        int movSiblingPosition = -1;
        boolean delPointer = false;
        List<Integer> cpyPointerParent = new ArrayList<>();
        List<Integer> cpySiblingPosition = new ArrayList<>();
        int id = tree.getId();
        if (delMap.keySet().contains(id) && delMap.get(id) == tree) {
            delPointer = true;
            action = DEL;
        }
        if (updatesMap.keySet().contains(id) && updatesMap.get(id).getValue0() == tree) {
            action = UPD;
            updPointer.addAll(updatesMap.get(id).getValue1());
        }
        if (movMap.keySet().contains(id) && movMap.get(id).getValue0() == tree) {
            action = MOV;
            movPointerParent = movMap.get(id).getValue1().getId();
            movSiblingPosition = movMap.get(id).getValue2();
        }
        if (cpyMap.keySet().contains(id) && cpyMap.get(id).getValue0() == tree) {
            action = INS;
            List<Insert> inserts = cpyMap.get(id).getValue1();
            cpyPointerParent = inserts.stream().map(Insert::getParent).map(ITree::getId).collect(Collectors.toList());
            cpySiblingPosition = inserts.stream().map(Insert::getPosition).collect(Collectors.toList());
        }
        Node node = new Node(tree, id, type, label, parent, action, updPointer, movPointerParent, movSiblingPosition, delPointer, cpyPointerParent, cpySiblingPosition);
        node.setChildIdx(0);
        if (!isLeaf) {
            // add dummy
            node.addChild(new Node(nextId.getAndInc(), DUMMY, DUMMY, node), 0);
            int i = 1;
            for (ITree child : tree.getChildren()) {
                Node nChild = createTreePreOrder(node, child, ctx, delMap, updatesMap, movMap, cpyMap, nextId, isContext, recordChildren);
                if (nChild != null) {
                    node.addChild(nChild);
                    i++;
                    if (recordChildren) {
                        if (!Node.legalChildMap.containsKey(nChild.type)) {
                            Node.legalChildMap.put(nChild.type, new HashSet<>());
                        }
                        Node.legalChildMap.get(nChild.type).add(type);
                    }
                }
            }
        }
        if (parent == null) {
            node.addChild(new Node(nextId.getAndInc(), DEL, DEL, node));
            if (isContext) {
                node.addChild(new Node(nextId.getAndInc(), INS, INS, node));
                node.addChild(new Node(nextId.getAndInc(), INS, UPD, node));
            }
        }
        return node;
    }

    private static ArrayList<Node> getTreeStack(Node node) {
        ArrayList<Node> upStack = new ArrayList<>();
        Node current = node;
        while (current != null) {
            upStack.add(current);
            current = current.getParentNode();
        }
        return upStack;
    }

    private ASTPath generatePath(Node source, Node target) {
        Pair<Integer, Integer> key = new Pair<>(source.id, target.id);
        if (pathsMap.containsKey(key)) {
            return pathsMap.get(key);
        }
        ASTPath path = new ASTPath(source.id, target.id);
        ArrayList<Node> sourceStack = getTreeStack(source);
        ArrayList<Node> targetStack = getTreeStack(target);

        int commonPrefix = 0;
        int currentSourceAncestorIndex = sourceStack.size() - 1;
        int currentTargetAncestorIndex = targetStack.size() - 1;
        while (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0
                && sourceStack.get(currentSourceAncestorIndex) == targetStack.get(currentTargetAncestorIndex)) {
            commonPrefix++;
            currentSourceAncestorIndex--;
            currentTargetAncestorIndex--;
        }

        for (int i = 0; i < sourceStack.size() - commonPrefix; i++) {
            Node currentNode = sourceStack.get(i);
            path.addNodeUp(currentNode);
        }

        Node commonNode = sourceStack.get(sourceStack.size() - commonPrefix);
        path.addNode(commonNode);

        for (int i = targetStack.size() - commonPrefix - 1; i >= 0; i--) {
            Node currentNode = targetStack.get(i);
            path.addNodeDown(currentNode);
        }
        pathsMap.putAll(path.getAllSubPathsMap());
        return path;
    }

    private static int getSizeNoDummies(Node node) {
        return getNodesList(node)
                .stream()
                .filter(n -> !n.getValue().equals(Node.DUMMY))
                .collect(Collectors.toList())
                .size();
    }

    private static List<Node> getNodesList(Node node) {
        List<Node> nodesList = new ArrayList<>();
        nodesList.add(node);
        if (node.children != null) {
            for (Node child : node.children) {
                nodesList.addAll(getNodesList(child));
            }
        }
        return nodesList;
    }

    private static List<Node> getNodesListLeavseFirst(Node node) {
        List<Node> nodesList = getNodesList(node);
        nodesList = nodesList.stream()
                .sorted((n1,n2)-> {
                            if (n1.isLeaf()) {
                                if (n2.isLeaf()) {
                                    return 0;
                                }
                                return -1;
                            }
                            return 1;

                        }).collect(Collectors.toList());
        return nodesList;
    }



    private static String extractOp(Node source, Node target, List<MyAction> actions) {
        int src = source.id;
        int tgt = target.id;
        List<String> ops = actions.stream().filter(a -> a.srcId == src && a.tgtId == tgt).map(MyAction::getName).collect(Collectors.toList());
        if (ops.size() == 0) {
            return NOP;
        }
        return ops.get(0);
    }

    private void extractPathsAux(List<ASTPath> pathsNOP, List<Triplet<ASTPath,Node,Node>> pathsOpTriplets, Node source, Node target, List<MyAction> actions) {
        ASTPath path = generatePath(source, target);
        if (!path.isEmpty()) {
            String op = extractOp(source, target, actions);
            path.setOp(op);
            if (op.equals(NOP)) {
                pathsNOP.add(path);
            } else {
                Triplet<ASTPath,Node,Node> t = new Triplet<>(path, source, target);
                pathsOpTriplets.add(t);
            }
        }
    }

    private static boolean relevantPath(Node source, Node target, Map<Node, Set<Node>> descendantMap) {
        if (source.value.equals(DEL)) {
            return false;
        }
        if (source.value.equals(DUMMY)) {
            return false;
        }
        if (source.parent == null || target.parent == null) {
            return false;
        }
        if (target.value.equals(DEL)) {
            return true;
        }
        //if (new HashSet<>(Node.getNodesList(source)).contains(target)) {
        if (!descendantMap.containsKey(source)) {
            descendantMap.put(source, new HashSet<>(Node.getNodesList(source)));
        }
        if (descendantMap.get(source).contains(target)) {
            return false;
        }
        // Not sure this is good:
//        if (source.parent.equals(target.parent) &&
//                source.parent.children.indexOf(source) - 1 == target.parent.children.indexOf(target) &&
//                (source.children.size() != 0 || target.children.size() != 0 )) {
//            return false;
//        }
        if (target.parent.parent == null) {
            return true;
        }
        if (Node.legalChildMap != null && !Node.legalChildMap.containsKey(source.type) ) {
            return true;
        }
        if (Node.legalChildMap != null &&
                Node.legalChildMap.get(source.type).contains(target.parent.type)) {
            return true;
        }
        return false;
    }

    public List<ASTPath> extractPaths(List<MyAction> actions, List<String> editScript, boolean keepNOP) {
        Node root = this;
        List<ASTPath>  pathsNOP = new ArrayList<>();
        List<Triplet<ASTPath,Node,Node>> pathsOpTriplets = new ArrayList<>();
        List<Node> nodesList = getNodesListLeavseFirst(root);
        Set<Pair<Integer,Integer>> actionPairSet = actions.stream().map(a -> new Pair<>(a.getSrcId(), a.getTgtId())).collect(Collectors.toSet());
        Map<Node, Set<Node>> descendantMap = new HashMap<>();
        for (int i = 0; i < nodesList.size(); i++) {
            for (int j = i + 1; j < nodesList.size(); j++) {
                Node source = nodesList.get(i);
                Node target = nodesList.get(j);
                if (!keepNOP) {
                    if (actionPairSet.contains(new Pair<>(source.getId(), target.getId()))) {
                        extractPathsAux(pathsNOP, pathsOpTriplets, source, target, actions);
                    }
                    if (actionPairSet.contains(new Pair<>(target.getId(), source.getId()))) {
                        extractPathsAux(pathsNOP, pathsOpTriplets, target, source, actions);
                    }
                } else {
                    if (relevantPath(source, target, descendantMap)) {
                        extractPathsAux(pathsNOP, pathsOpTriplets, source, target, actions);
                    }
                    if (relevantPath(target, source, descendantMap)) {
                        extractPathsAux(pathsNOP, pathsOpTriplets, target, source, actions);
                    }
                }
            }
        }
        List<MyAction> actionsReversed = new ArrayList<>(actions);
        Collections.reverse(actionsReversed);
        List<ASTPath> pathsOp = new LinkedList<>();
        for (MyAction action : actionsReversed) {
            int src = action.getSrcId();
            int tgt = action.getTgtId();
            pathsOp.addAll(0,
                    pathsOpTriplets.stream()
                            .filter(t -> t.getValue1().id == src &&
                                    t.getValue2().id == tgt)
                            .map(Triplet::getValue0)
                            .collect(Collectors.toList()));
        }
        for (ASTPath p : pathsOp) {
            String op = p.getOp();
            int sourceId = p.getSourceId();
            int targetId = p.getTargetId();
            if (op.equals(DEL)) {
                editScript.add(op + " " + sourceId);
            } else {
                editScript.add(op + " " + sourceId + " " + targetId);
            }

        }
        if (keepNOP) {
            pathsOp.addAll(pathsNOP);
        }
        return pathsOp;
    }

    private static void toDotStringRec(Node node, List<String> nodes, List<String> edges) {
        String value = node.value;
        value = value.replaceAll("<", "\\\\<").replaceAll(">", "\\\\>").replaceAll("\\|", "\\\\|");
        nodes.add(node.id + " [label=\"{" + node.id + "|" + value + "}\"];");
        for (Node child : node.children) {
            edges.add(node.id + " -> " + child.id + ";");
            toDotStringRec(child, nodes, edges);
        }
    }

    public static String toDotString(Node node) {
        List<String> nodes = new ArrayList<>();
        List<String> edges = new ArrayList<>();
        toDotStringRec(node, nodes, edges);
        StringBuilder stringBuilder = new StringBuilder(SEP);
        stringBuilder.append("strict digraph tree {");
        stringBuilder.append("node [shape=record fontname=Arial];");
        stringBuilder.append(String.join(SEP, nodes));
        stringBuilder.append(String.join(SEP, edges));
        stringBuilder.append("}");
        return stringBuilder.toString();
    }

    private static boolean toStringASTRec(Node node, StringJoiner builder) {
        if (node.value.equals(DUMMY) || node.value.equals(DEL)) {
            return false;
        }
        if (node.children.size() > 0) {
            builder.add(node.value);
            builder.add("(");
            for (int i = 0; i < node.children.size(); i++) {
                boolean added = toStringASTRec(node.children.get(i), builder);
                if (added && i < node.children.size() - 1 && !node.children.get(i+1).value.equals(DEL)) {
                    builder.add(",");
                }
            }
            builder.add(")");
        } else {
            builder.add("(");
            List<String> nameParts = Common.splitToSubtokens(node.value);
            builder.add(String.join(" , ", nameParts));
            builder.add(")");
        }

        return true;
    }

    public static String toStringAST(Node node) {
        StringJoiner builder = new StringJoiner(SEP);
        toStringASTRec(node, builder);
        return builder.toString();
    }




    public static void createIdTokenList(Node node, List<Pair<Integer,String>> idToken) {
        String value = node.getValue();
        int id = node.getId();
        idToken.add(new Pair<>(id, value));
        if (!node.isLeaf()) {
            for(Node child : node.getChildren()) {
                createIdTokenList(child, idToken);
            }
        }
    }

    public static void createValueNodeMap(ITree tree, Map<String, List<Integer>> valueNodesMap) {
        String value = tree.getLabel();
        int id = tree.getId();
        if (valueNodesMap.keySet().contains(value)) {
            valueNodesMap.get(value).add(id);
        } else {
            valueNodesMap.put(value, new ArrayList<>(Arrays.asList(id)));
        }
        if (!tree.isLeaf()) {
            for(ITree child : tree.getChildren()) {
                createValueNodeMap(child, valueNodesMap);
            }
        }
    }

    @Override
    public String toString() {
        return this.value;
    }
}
