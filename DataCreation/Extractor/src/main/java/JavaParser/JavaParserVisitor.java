///*
// * This file is part of GumTree.
// *
// * GumTree is free software: you can redistribute it and/or modify
// * it under the terms of the GNU Lesser General Public License as published by
// * the Free Software Foundation, either version 3 of the License, or
// * (at your option) any later version.
// *
// * GumTree is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU Lesser General Public License for more details.
// *
// * You should have received a copy of the GNU Lesser General Public License
// * along with GumTree.  If not, see <http://www.gnu.org/licenses/>.
// *
// * Copyright 2018 Jean-Rémy Falleri <jr.falleri@gmail.com>
// */
//
//package JavaParser;
//
//import com.github.gumtreediff.io.LineReader;
//import com.github.gumtreediff.tree.ITree;
//import com.github.gumtreediff.tree.TreeContext;
//import Extractor.Utils.Type;
//import Extractor.Utils.TypeSet;
//import com.github.javaparser.Position;
//import com.github.javaparser.ast.Node;
//import com.github.javaparser.ast.expr.*;
//import com.github.javaparser.ast.visitor.TreeVisitor;
//
//import java.util.ArrayDeque;
//import java.util.ArrayList;
//import java.util.Deque;
//import java.util.NoSuchElementException;
//
//public class JavaParserVisitor extends TreeVisitor {
//
//    protected TreeContext context;
//
//    private Deque<ITree> trees;
//
//    private LineReader reader;
//
//    public JavaParserVisitor(LineReader reader) {
//        this.context = new TreeContext();
//        this.trees = new ArrayDeque<>();
//        this.reader = reader;
//    }
//
//    public TreeContext getTreeContext() {
//        return context;
//    }
//
//
//    public void visitPreOrder(Node node) {
//        process(node);
//        new ArrayList<>(node.getChildrenNodes()).forEach(this::visitPreOrder);
//        if (trees.size() > 0)
//            trees.pop();
//    }
//
//    @Override
//    public void process(Node node) {
//        String label = node.getClass().getSimpleName();
//        String operator = "";
//        if (node instanceof BinaryExpr) {
//            operator = ((BinaryExpr) node).getOperator().toString();
//        } else if (node instanceof UnaryExpr) {
//            operator = ((UnaryExpr) node).getOperator().toString();
//        } else if (node instanceof AssignExpr) {
//            operator = ((AssignExpr) node).getOperator().toString();
//        } else if (node instanceof NameExpr) {
//            operator = node.toString();
//        } else if (node instanceof VariableDeclarationExpr) {
//            operator = node.toString();
//        } else if (node instanceof LiteralExpr) {
//            operator = node.toString();
//        }
//
//        if (operator.length() > 0) {
//            label += ":" + operator;
//        }
//
//
////        if (node instanceof SimpleName)
////            label = ((SimpleName) node).getIdentifier();
////        else if (node instanceof StringLiteralExpr)
////            label = ((StringLiteralExpr) node).asString();
////        else if (node instanceof BooleanLiteralExpr)
////            label = Boolean.toString(((BooleanLiteralExpr) node).getValue());
////        else if (node instanceof LiteralStringValueExpr)
////            label = ((LiteralStringValueExpr) node).getValue();
////        else if (node instanceof PrimitiveType)
////            label = ((PrimitiveType) node).asString();
////        else if (node instanceof Modifier)
////            label = ((Modifier) node).getKeyword().asString();
//        if (!node.toString().isEmpty() && !"null".equals(node.toString())) {
//            pushNode(node, label);
//        }
//    }
//
//    protected void pushNode(Node n, String label) {
//        try {
//            Position begin = n.getRange().begin;
//            Position end = n.getRange().end;
//            int startPos = reader.positionFor(begin.line, begin.column);
//            int length = reader.positionFor(end.line, end.column) - startPos + 2;
//            push(nodeAsSymbol(n), label, startPos, length);
//        }
//        catch (NoSuchElementException ignore) { }
//
//    }
//
//    protected Type nodeAsSymbol(Node n) {
//        return TypeSet.type(n.getClass().getSimpleName());
//    }
//
//    private void push(Type type, String label, int startPosition, int length) {
//        ITree t = context.createTree(type.type_number, label, type.toString());
//        t.setPos(startPosition);
//        t.setLength(length);
//
//        if (trees.isEmpty())
//            context.setRoot(t);
//        else {
//            ITree parent = trees.peek();
//            t.setParentAndUpdateChildren(parent);
//        }
//
//        trees.push(t);
//    }
//}
