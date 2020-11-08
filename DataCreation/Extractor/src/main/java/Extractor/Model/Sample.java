package Extractor.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Sample {
    private List<ASTPath> source;
    private List<String> target;
    private List<String> editScript;
    private List<String> idToken;
    private String dotString;
    private String pathsDotString;
    private String ASTbefore;
    private String ASTafter;

    public Sample(List<ASTPath> source, List<String> target, List<String> editScript,
                  List<String> idToken, String dotString, String pathsDotString,
                  String ASTbefore, String ASTafter) {
        this.source = source;
        this.target = target;
        this.editScript = editScript;
        this.idToken = idToken;
        this.dotString = dotString;
        this.pathsDotString = pathsDotString;
        this.ASTbefore = ASTbefore;
        this.ASTafter = ASTafter;
    }

    public Sample() {
        this.source = new ArrayList<>();
        this.target = new ArrayList<>();
        this.editScript = new ArrayList<>();
        this.idToken = new ArrayList<>();
        this.dotString = "";
        this.pathsDotString = "";
        this.ASTbefore = "";
        this.ASTafter = "";
    }

    public List<ASTPath> getSource() {
        return source;
    }

    public String getSourceAsString(String sep) {
        return source.stream().map(ASTPath::toString).collect(Collectors.joining(sep));
    }

    public String getSourceIds(String sep) {
        return source.stream().map(p -> p.getSourceId() + " " + p.getTargetId()).collect(Collectors.joining(sep));
    }

    public String getPathsOp(String sep) {
        return source.stream().map(ASTPath::FormattedOpString).collect(Collectors.joining(sep));
    }

    public int getNumOfPaths() {
        return source.size();
    }

    public String getTargetAsString(String sep) {
        return String.join(sep, target);
    }

    public String getEditScriptAsString(String sep) {
        return String.join(sep, editScript);
    }

    public String getIdToeknAsString(String sep) {
        return String.join(sep, idToken);
    }

    public List<String> getTarget() {
        return target;
    }

    public List<String> getEditScript() {
        return editScript;
    }
    public String getDotString() {
        return this.dotString;
    }
    public String getPathDotString() { return this.pathsDotString; }
    public String getASTbeforeString() { return this.ASTbefore; }
    public String getASTafterString() { return this.ASTafter; }
}
