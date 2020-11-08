package Extractor.Utils;

import Extractor.Model.MyAction;
import Extractor.Model.Node;
import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeUtils;

import java.util.*;
import java.util.stream.Collectors;

public class ActionUtil {
    private ActionUtil() {
    }

    public static void apply(List<Action> actions) {
        Iterator var2 = actions.iterator();

        while(var2.hasNext()) {
            Action a = (Action)var2.next();
            if (a instanceof Insert) {
                Insert action = (Insert)a;
                ITree toInsert = action.getNode();
                toInsert.setChildren(new ArrayList<>());
                action.getParent().insertChild(toInsert, action.getPosition());
            } else if (a instanceof Update) {
                Update action = (Update)a;
                action.getNode().setLabel(action.getValue());
            } else if (a instanceof Move) {
                Move action = (Move)a;
                if (action.getNode().getParent() == action.getParent()) {
                    action = new Move(action.getNode(), action.getParent(), action.getPosition() - 1);
                }
                action.getNode().getParent().getChildren().remove(action.getNode());
                action.getParent().insertChild(action.getNode(), action.getPosition());
            } else {
                if (!(a instanceof Delete)) {
                    throw new RuntimeException("No such action: " + a);
                }
                Delete action = (Delete)a;
                action.getNode().getParent().getChildren().remove(action.getNode());
            }
        }
    }

    private static List<Action> reduceDelActions(List<Action> actions) {
        List<ITree> delNodes = actions.stream().filter(x -> x instanceof Delete)
                .map(Action::getNode)
                .collect(Collectors.toList());
        Collections.reverse(delNodes);
        Set<ITree> delSet = new HashSet<>(delNodes);
        Set<ITree> removed = new HashSet<>();
        for(final ITree node : delNodes) {
            if (delSet.contains(node.getParent())) {
                removed.add(node);
            }
        }
        List<Action> reduced = new ArrayList<>();
        for (Action action: actions) {
            if (action instanceof Delete) {
                Delete delAction = (Delete)action;
                if (removed.contains(delAction.getNode())) {
                    continue;
                }
            }
            reduced.add(action);
        }
        return reduced;
    }

    private static List<Action> reduceMovActions(List<Action> actions) {
        List<ITree> delNodes = actions.stream().filter(x -> x instanceof Delete)
                .map(Action::getNode)
                .collect(Collectors.toList());
        Set<Integer> removed = new HashSet<>();
        for(final ITree node : delNodes) {
            List<Integer> nodes = TreeUtils.breadthFirst(node).stream().map(ITree::getId).collect(Collectors.toList());
            removed.addAll(nodes);
        }
        List<ITree> movNodes = actions.stream().filter(x -> x instanceof Move)
                .map(Action::getNode)
                .collect(Collectors.toList());
        Set<Integer> moved = new HashSet<>();
        for(final ITree node : movNodes) {
            List<Integer> nodes = TreeUtils.breadthFirst(node).stream().map(ITree::getId).collect(Collectors.toList());
            moved.addAll(nodes);
        }
        List<Action> reduced = new ArrayList<>();
        for (Action action: actions) {
            if (action instanceof Move) {
                Move movAction = (Move)action;
                if (removed.contains(movAction.getParent().getId()) && removed.contains(movAction.getNode().getId())
                && !moved.contains(movAction.getParent().getId())) {
                    continue;
                }
            }
            reduced.add(action);
        }
        return reduced;
    }

    private static List<Action> reduceMovDelActions(List<Action> actions) {
        List<Action> delActions = actions.stream().filter(x -> x instanceof Delete)
                .collect(Collectors.toList());
        Set<Action> delSet = new HashSet<>(delActions);
        Set<Action> removedActions = new HashSet<>();
        for (Action action: actions) {
            if (action instanceof Move) {
                Move movAction = (Move) action;
                ITree movNode = movAction.getNode();
                ITree parent = movAction.getParent();
                Set<Action> delChildren = delSet.stream().filter(x -> x.getNode().getParent().equals(parent)).collect(Collectors.toSet());
                int pos = movAction.getPosition();
                ITree rightSibling = null;
                ITree leftSibling = null;
                if (pos < parent.getChildren().size()) {
                    rightSibling = parent.getChild(pos);
                }
                if (pos > 0 && pos - 1 < parent.getChildren().size()) {
                    leftSibling = parent.getChild(pos - 1);
                }
                for (Action a : delChildren) {
                    if ((rightSibling != null && rightSibling.equals(a.getNode()) && rightSibling.isIsomorphicTo(movNode)) ||
                            (leftSibling != null && leftSibling.equals(a.getNode()) && leftSibling.isIsomorphicTo(movNode))) {
                        removedActions.add(movAction);
                        removedActions.add(a);
                    }
                }
            }
        }
        actions.removeAll(removedActions);
        return actions;
    }


    private static List<Action> reduceInsertActions(List<Action> actions) {
        List<ITree> insertNodes = actions.stream().filter(x -> x instanceof Insert)
                .map(Action::getNode)
                .collect(Collectors.toList());
        Set<ITree> InsertSet = new HashSet<>(insertNodes);
        Set<ITree> inserted = new HashSet<>();
        for(final ITree node : insertNodes) {
            List<ITree> nodes = TreeUtils.breadthFirst(node);
            nodes.stream().filter(x -> x != node)
                    .filter(x -> InsertSet.contains(x))
                    .forEach(x -> inserted.add(x));
        }
        List<Action> reduced = new ArrayList<>();
        for (Action action: actions) {
            if (action instanceof Insert) {
                Insert insertAction = (Insert)action;
                if (inserted.contains(insertAction.getNode())) {
                    continue;
                }
            }
            reduced.add(action);
        }
        return reduced;
    }


    public static List<Action> reduceActions(List<Action> actions, ITree node, Map<Action, ITree> InsertToNode) {
//        apply(actions);
        actions = reduceDelActions(actions);
        actions = reduceInsertActions(actions);
        actions = reduceMovActions(actions);
        actions = reduceMovDelActions(actions);
        // Check if Ins operation is actualy a Copy opeartion: foo(x+2,y+3);bar(m); -> foo(x+2,y+3,m);bar(m);
        getInsertToNodeMap(actions, node, InsertToNode);
        updateInsertNodes(actions, InsertToNode);
        actions = changeInsToMov(actions);
        return actions;
    }

    public static void getInsertToNodeMap(List<Action> actions, ITree node, Map<Action, ITree> InsertToNode) {
        List<ITree> nodes = TreeUtils.breadthFirst(node);
        for (Action action: actions) {
            if (action instanceof Insert) {
                ITree actionNode = action.getNode();
                List<ITree> isomprohicList = nodes.stream().filter(n -> n.isIsomorphicTo(actionNode)).collect(Collectors.toList());
                if (isomprohicList.size() > 0) {
                    InsertToNode.put(action, isomprohicList.get(0));
                }
            }
        }
    }

    public static void updateInsertNodes(List<Action> actions, Map<Action, ITree> insertToNode) {
        for(Action action : actions) {
            if (action instanceof Insert && insertToNode.containsKey(action)) {
                action.setNode(insertToNode.get(action));
            }
        }
    }
    public static List<Action> changeInsToMov(List<Action> actions) {
        List<ITree> insertNodes = actions.stream().filter(x -> x instanceof Insert)
                .map(Action::getNode)
                .collect(Collectors.toList());
        List<ITree> movNodes = actions.stream().filter(x -> x instanceof Move)
                .map(Action::getNode)
                .collect(Collectors.toList());
        Set<ITree> movSet = new HashSet<>(movNodes);
        Set<ITree> InsertSet = new HashSet<>();
        Set<ITree> dupInsertSet = new HashSet<>();
        for (ITree node : insertNodes)
        {
            if (!InsertSet.add(node))
            {
                dupInsertSet.add(node);
            }
        }
        InsertSet.removeAll(dupInsertSet);
        List<ITree> delNodes = actions.stream().filter(x -> x instanceof Delete)
                .map(Action::getNode)
                .collect(Collectors.toList());
        Set<ITree> removed = new HashSet<>();
        for(final ITree node : delNodes) {
            List<ITree> nodes = TreeUtils.breadthFirst(node);
            nodes.stream().filter(x -> InsertSet.contains(x))
                    .forEach(x -> removed.add(x));
        }
        List<Action> reduced = new ArrayList<>();
        Map<ITree, List<Integer>> parentRemoved = new HashMap<>();
        Action action = null;
        for (Iterator i = actions.iterator(); i.hasNext(); ) {
            action =(Action) i.next();
            if (action == null) {
                continue;
            }
            if (action instanceof Insert) {
                Insert insertAction = (Insert)action;
                ITree insertNode = insertAction.getNode();
                if (removed.contains(insertNode) && !movSet.contains(insertNode)) {
                    int pos = insertAction.getPosition();
                    int toReduce = 0;
                    ITree parentNode = insertAction.getParent();
                    if (parentRemoved.containsKey(parentNode)) {
                        List<Integer> removedChildren = parentRemoved.get(parentNode);
                        for (Integer j : removedChildren) {
                            if( j < pos) {
                                toReduce++;
                            }
                        }
                    }
                    pos -= toReduce;
                    reduced.add(new Move(insertAction.getNode(), insertAction.getParent(), pos));
                    if (delNodes.contains(insertNode)) {
                        Action delAction = actions.stream()
                                .filter(a -> a != null)
                                .filter(a -> a.getNode().equals(insertNode))
                                .filter(a -> a instanceof Delete)
                                .collect(Collectors.toList()).get(0);
                        actions.set(actions.indexOf(delAction), null);
//                        actions.remove(delAction);
                        parentNode = delAction.getNode().getParent();
                        if (!parentRemoved.containsKey(parentNode)) {
                            parentRemoved.put(parentNode, new ArrayList<>());
                        }
                        parentRemoved.get(parentNode).add(delAction.getNode().positionInParent());
                    }
                } else {
                    reduced.add(action);
                }
            } else {
                reduced.add(action);
            }
        }
        return reduced;
    }

}
