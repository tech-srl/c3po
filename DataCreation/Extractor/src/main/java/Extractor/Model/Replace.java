package Extractor.Model;

import com.github.gumtreediff.actions.model.Action;
import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;

public class Replace extends Action {
    private ITree node_2;

    public Replace(ITree node, ITree node_2) {

        super(node);
        this.node_2 = node_2;
    }

    public String getName() {
        return "REPLACE";
    }

    public String toString() {
        return this.getName() + " " + this.node.toShortString() + " with" + this.node_2;
    }

    public String format(TreeContext ctx) {
        return this.getName() + " " + this.node.toPrettyString(ctx) + " with" + this.node_2.toPrettyString(ctx);
    }
}
