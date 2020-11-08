package Extractor;

import Extractor.Common.Common;
import Extractor.Model.*;
import Extractor.Utils.ActionUtil;
import Extractor.Utils.CommandLineValues;
import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.actions.model.*;
import com.github.gumtreediff.gen.srcml.SrcmlCsTreeGenerator;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import com.github.gumtreediff.tree.TreeUtils;
import org.javatuples.Pair;
import org.javatuples.Triplet;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

class ParsingTask implements Callable<TreeContext> {
    private String code;
    private int tries;
    private SrcmlCsTreeGenerator generator;

    public ParsingTask(String code, int tries ,SrcmlCsTreeGenerator generator) {
        this.code = code;
        this.tries = tries;
        this.generator = generator;
    }

    @Override
    public TreeContext call() {
        TreeContext context;
        while(true) {
            try {
                context = generator.generateFromString(code);
            } catch (RuntimeException e) {
                if (tries == 0) {
                    return null;
                }
                tries--;
                continue;
            } catch (Exception e) {
                return null;
            }
            break;
        }
        return context;
    }
}


public class ExecutorTask implements Callable<Triplet<Sample,Sample,Sample>> {

    public static String currentProject = "";
    public static AtomicInteger total = new AtomicInteger();
    public static AtomicInteger progress = new AtomicInteger();
    public static AtomicInteger failed = new AtomicInteger();
    public static AtomicInteger noActions = new AtomicInteger();
    public static AtomicInteger invalidActions = new AtomicInteger();
    public static AtomicInteger nodeExceeded = new AtomicInteger();

    public static AtomicLong delActions = new AtomicLong();
    public static AtomicLong movActions = new AtomicLong();
    public static AtomicLong updActions = new AtomicLong();
    public static AtomicLong insActions = new AtomicLong();

    public static AtomicLong insSubTreeSizes = new AtomicLong();
    public static AtomicLong movSubTreeSizes = new AtomicLong();
    public static AtomicLong delSubTreeSizes = new AtomicLong();

    private static int PARSE_TRIES = 10;
    private static int PARSE_TIMEOUT = 10;

    public static boolean recordChildren = false;

    public static CommandLineValues commandLineValues;
    private Pair<String,String> beforeAfterPair;
    private Pair<String,String> beforeContextBeforeAfterPair;
    private Pair<String,String> afterContextBeforeAfterPair;


    public ExecutorTask(CommandLineValues commandLineValues, Pair<String, String> beforeAfterPair, Pair<String, String> beforeContextBeforeAfterPair, Pair<String, String> afterContextBeforeAfterPair) {
        ExecutorTask.commandLineValues = commandLineValues;
        this.beforeAfterPair = beforeAfterPair;
        this.beforeContextBeforeAfterPair = beforeContextBeforeAfterPair;
        this.afterContextBeforeAfterPair = afterContextBeforeAfterPair;
    }

    public static void initialCounters(int total, String projectName){
        ExecutorTask.progress.set(0);
        ExecutorTask.failed.set(0);
        ExecutorTask.noActions.set(0);
        ExecutorTask.invalidActions.set(0);
        ExecutorTask.nodeExceeded.set(0);
        ExecutorTask.total.set(total);
        ExecutorTask.currentProject = projectName;
    }

    @Override
    public Triplet<Sample,Sample,Sample> call() {
        Sample focus = process(this.beforeAfterPair, false);
        if (focus == null) {
            return null;
        }
        if (recordChildren) {
            return null;
        }
        Sample beforeContext = process(this.beforeContextBeforeAfterPair, true);
        int beforeContextNumPaths = beforeContext == null ? 0 : beforeContext.getNumOfPaths();
        if (beforeContextNumPaths > commandLineValues.maxPathCtx) {
            return null;
        }

        Sample afterContext = process(this.afterContextBeforeAfterPair, true);
        int afterContextNumPaths = afterContext == null ? 0 : afterContext.getNumOfPaths();
        if (beforeContextNumPaths + afterContextNumPaths > commandLineValues.maxPathCtx) {
            return null;
        }
        if (beforeContext == null && afterContext == null) {
            return null;
        }
        System.out.print("\r" + currentProject + ": " + progress.incrementAndGet() + "/" + total);
        return new Triplet<>(focus, beforeContext, afterContext);
    }

    private static String fixNumStr(String code) {
        String res = code.replaceAll("NUM", "999");
        res = res.replaceAll("STR", "\"STR\"");
        return res;
    }

    private static void fixNumStr(ITree tree, TreeContext ctx) {
        List<ITree> nodes = TreeUtils.breadthFirst(tree);
        for (ITree node : nodes) {
            if (ctx.getTypeLabel(node.getType()).equals("literal")) {
                String newLabel = Node.fixLiteral(node.getLabel());
                switch(newLabel) {
                        case Node.CHAR_LITERAL:
                            newLabel = "'c'";
                            break;
                        case Node.STRING_LITERAL:
                            newLabel = "\"" + Node.STRING_LITERAL + "\"";
                            break;
                        default:
                            newLabel = node.getLabel();
                }
                node.setLabel(newLabel);
            }
        }
    }

    private static TreeContext parseCode(String code, SrcmlCsTreeGenerator generator) throws Exception {
        // redirect all GumTree print outputs
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.err;
        System.setErr(ps);

        TreeContext result;
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<TreeContext> future = executor.submit(new ParsingTask(code, PARSE_TRIES, generator));
            result = future.get(PARSE_TIMEOUT, TimeUnit.SECONDS);
        } catch (Exception e) {
            throw e;
        } finally {
            executor.shutdownNow();
            System.err.flush();
            System.setErr(old);
        }
        return result;

    }



    public static Sample process(Pair<String, String> pair, boolean isContext) {
        SrcmlCsTreeGenerator generator = new SrcmlCsTreeGenerator();
        String before = pair.getValue0().replaceAll("\\\\n", "\n");
        String after = pair.getValue1().replaceAll("\\\\n", "\n");
        before = fixNumStr(before);
        after = fixNumStr(after);
        TreeContext beforeContext;
        TreeContext afterContext;
        try {
            beforeContext = parseCode(before, generator);
            afterContext = parseCode(after, generator);
            if (beforeContext == null || afterContext == null) {
                throw new Exception();
            }
        } catch (Exception e) {
            failed.incrementAndGet();
            return null;
        }
        afterContext.importTypeLabels(beforeContext);
        beforeContext.importTypeLabels(afterContext);
        ITree beforeTree = beforeContext.getRoot();
        ITree afterTree = afterContext.getRoot();
        fixNumStr(beforeTree, beforeContext);
        fixNumStr(afterTree, afterContext);
        Matcher m = Matchers.getInstance().getMatcher(beforeTree, afterTree);
        m.match();
        ActionGenerator g = new ActionGenerator(beforeTree, afterTree, m.getMappings());
        g.generate();
        Map<Action, ITree> InsertToNode = new HashMap<>();
        List<Action> actions = ActionUtil.reduceActions(g.getActions(), beforeTree, InsertToNode);
        if (actions.size() == 0 && !commandLineValues.inferenceSample) {
            failed.incrementAndGet();
            return null;
        }
        Map<String, List<Integer>> valueNodesMap = new HashMap<>();
        Node.createValueNodeMap(beforeTree, valueNodesMap);

        Map<Integer, Pair<ITree,List<Integer>>> updatesMap = actions.stream()
                .filter(action -> action instanceof Update)
                .filter(action -> valueNodesMap.containsKey(((Update) action).getValue()))
                .collect(Collectors.toMap(action -> action.getNode().getId(), action -> new Pair<>(action.getNode(), valueNodesMap.get(((Update) action).getValue()))));
        Map<Integer, ITree> delMap = actions.stream()
                .filter(action -> action instanceof Delete)
                .collect(Collectors.toMap(action -> action.getNode().getId(), Action::getNode));
        Map<Integer, Triplet<ITree,ITree,Integer>> movMap = actions.stream()
                .filter(action -> action instanceof Move)
                .collect(Collectors.toMap(action -> action.getNode().getId(), action -> new Triplet<>(action.getNode(), ((Move) action).getParent(), ((Move) action).getPosition())));

        Map<Integer, Pair<ITree,List<Insert>>> cpyMap = new HashMap<>();
        int validIns = 0;
        for(Action action : actions) {
            if (action instanceof Insert && InsertToNode.containsKey(action)) {
                int key = action.getNode().getId();
                if (!cpyMap.containsKey(key)) {
                    cpyMap.put(key, new Pair<>(action.getNode(), new ArrayList<>()));
                }
                cpyMap.get(key).getValue1().add((Insert) action);
                validIns += 1;
            }
        }

        long numMovs = actions.stream().filter(action -> action instanceof Move).count();
        long numUpds = actions.stream().filter(action -> action instanceof Update).count();
        long numDels = actions.stream().filter(action -> action instanceof Delete).count();
        long numIns = actions.stream().filter(action -> action instanceof Insert).count();

//        if (!isContext) {
//            delActions.addAndGet(numDels);
//            movActions.addAndGet(numMovs);
//            updActions.addAndGet(numUpds);
//            insActions.addAndGet(numIns);
//            System.out.print("\r" + currentProject + ": " + progress.incrementAndGet() + "/" + total);
//        }

        if (!isContext && !commandLineValues.inferenceSample) {
            if ((commandLineValues.noINS && numIns > 0) ||
                    (!commandLineValues.noINS && numIns != validIns) ||
                    numMovs != movMap.size() ||
                    numUpds != updatesMap.size()) {
                invalidActions.incrementAndGet();
                return null;
            }
        }

        List<Integer> ids = valueNodesMap.values().stream().flatMap(List::stream).collect(Collectors.toList());
        int maxId = Collections.max(ids);
        int numOfNodes = beforeTree.getSize();
        if (!isContext && numOfNodes > commandLineValues.maxNodes && !commandLineValues.inferenceSample) {
            nodeExceeded.incrementAndGet();
            return null;
        }

        if(!isContext && !commandLineValues.inferenceSample && updatesMap.isEmpty() && movMap.isEmpty() && cpyMap.isEmpty()) {
            noActions.incrementAndGet();
            return null;
        }

        MutableInteger nextId = new MutableInteger(maxId + 1);
        Node simpleTree;
        Node afterSimpleTree;
        try {
            simpleTree = Node.createTreePreOrder(null, beforeTree, beforeContext, delMap, updatesMap, movMap, cpyMap, nextId, isContext, recordChildren);
            afterSimpleTree = Node.createTreePreOrder(null, afterTree, afterContext, delMap, updatesMap, movMap, cpyMap, nextId, isContext, recordChildren);
        } catch (IllegalStateException e) {
            failed.incrementAndGet();
            return null;
        }
        if (recordChildren) {
            return null;
        }
        List<MyAction> myActions = null;
        try {
            myActions = Node.createMyActions(simpleTree, beforeTree, actions, isContext, beforeContext, nextId);
        } catch (IndexOutOfBoundsException e) {
        }

        if (myActions == null) {
            invalidActions.incrementAndGet();
            return null;
        }
        if (myActions.stream().filter(a -> a.getSrcId() == a.getTgtId()).collect(Collectors.toList()).size() > 0) {
            invalidActions.incrementAndGet();
            return null;
        }

        // Comment this!
//        if (!isContext) {
//            for(MyAction a : myActions) {
//                if (a instanceof MyMove) {
//                    movActions.addAndGet(1);
//                    movSubTreeSizes.addAndGet(a.getSubTreeSize());
//                } else if (a instanceof MyInsert) {
//                    insActions.addAndGet(1);
//                    insSubTreeSizes.addAndGet(a.getSubTreeSize());
//                } else if (a instanceof MyDelete) {
//                    delActions.addAndGet(1);
//                    delSubTreeSizes.addAndGet(a.getSubTreeSize());
//                }
//            }
//            return null;
//        }

        Node.clearCommentNodes(simpleTree);
        Node.clearCommentNodes(afterSimpleTree);

        List<String> editScriptList = new ArrayList<>();
        List<ASTPath> pathList = simpleTree.extractPaths(myActions, editScriptList, !isContext);
        if (editScriptList.size() == 0 && !commandLineValues.inferenceSample) {
            noActions.incrementAndGet();
            return null;
        }

        if (editScriptList.size() != myActions.size()) {
//            System.out.println("");
//            System.out.println(before);
//            System.out.println("+++++++++++++++++++");
//            System.out.println(after);
//            System.out.println("-------------------");
            return null;
        }

        String ASTbefore = Node.toStringAST(simpleTree);
        String ASTafter = Node.toStringAST(afterSimpleTree);
        String dotString = Node.toDotString(simpleTree);
        String pathsDotString = createPathDotString(pathList);
        List<Pair<Integer, String>> idTokenPairs = new ArrayList<>();
        Node.createIdTokenList(simpleTree, idTokenPairs);
        List<String> idToken = idTokenPairs.stream()
                .sorted(Comparator.comparingInt(Pair::getValue0))
                .map(p -> p.getValue0() + " " + p.getValue1())
                .collect(Collectors.toList());
        List<String> target = new ArrayList<>();
        int i = 0;
        // it's possible that only DEL remains
        boolean only_del = true;
        for (ASTPath p : pathList) {
            String op = p.getOp();
            if (op.equals(Common.NOP)) {
                break;
            }
            if (!op.equalsIgnoreCase(Common.DEL)) {
                only_del = false;
            }

            if (op.equals(Common.DEL)) {
                op = Common.MOV;
            }
            target.add(op + " " + i++);
        }
        if (!isContext && !commandLineValues.inferenceSample && only_del) {
            noActions.incrementAndGet();
            return null;
        }
        return new Sample(pathList, target, editScriptList, idToken, dotString, pathsDotString, ASTbefore, ASTafter);
    }

    private static String createPathDotString(List<ASTPath> paths) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("digraph G {");
        stringBuilder.append("node [shape=record fontname=Arial];");
        int id = 0;
        for (ASTPath path : paths) {
            stringBuilder.append(path.getDotStringFormat(id));
            id += path.getSize();
        }
        stringBuilder.append("}");
        return stringBuilder.toString();
    }
}
