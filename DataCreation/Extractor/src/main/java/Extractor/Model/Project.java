package Extractor.Model;

import Extractor.Common.Common;
import org.javatuples.Pair;
import org.javatuples.Triplet;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class Project {
    private String beforeFilePath;
    private String afterFilePath;
    private String beforeContextPath;
    private String afterContextPath;
    private String beforeNormalizedPath;
    private String afterNormalizedPath;
    private String beforeContextBeforePath;
    private String beforeContextAfterPath;
    private String afterContextBeforePath;
    private String afterContextAfterPath;
    private String beforeContextBeforeNormalizedPath;
    private String beforeContextAfterNormalizedPath;
    private String afterContextBeforeNormalizedPath;
    private String afterContextAfterNormalizedPath;
    private String integratedChangePath;

    private String projectDir;
    private String projectName;

    private List<String> beforeLines;
    private List<String> afterLines;
    private List<String> beforeContextBeforeLines;
    private List<String> beforeContextAfterLines;
    private List<String> afterContextBeforeLines;
    private List<String> afterContextAfterLines;

    public Project(String projectDir, String projectName) throws IOException{
        beforeFilePath = projectDir + "before.txt";
        afterFilePath = projectDir + "after.txt";
        beforeContextPath = projectDir + projectName + ".before_ctx";
        afterContextPath = projectDir + projectName + ".after_ctx";
        beforeNormalizedPath = projectDir + projectName + ".before_normalized";
        afterNormalizedPath = projectDir + projectName + ".after_normalized";
        beforeContextBeforePath = projectDir + projectName + ".before_ctx_before";
        beforeContextAfterPath = projectDir + projectName + ".before_ctx_after";
        afterContextBeforePath = projectDir + projectName + ".after_ctx_before";
        afterContextAfterPath = projectDir + projectName + ".after_ctx_after";
        beforeContextBeforeNormalizedPath = projectDir + projectName + ".before_ctx_before_normalized";
        beforeContextAfterNormalizedPath = projectDir + projectName + ".before_ctx_after_normalized";
        afterContextBeforeNormalizedPath = projectDir + projectName + ".after_ctx_before_normalized";
        afterContextAfterNormalizedPath = projectDir + projectName + ".after_ctx_after_normalized";
        integratedChangePath = projectDir + projectName + ".integrated_change";
        this.projectDir = projectDir;
        this.projectName = projectName;
        handleMissingFiles();
        beforeLines = Files.readAllLines(Paths.get(beforeFilePath));
        afterLines = Files.readAllLines(Paths.get(afterFilePath));
        beforeContextBeforeLines = Files.readAllLines(Paths.get(beforeContextBeforePath));
        beforeContextAfterLines = Files.readAllLines(Paths.get(beforeContextAfterPath));
        afterContextBeforeLines = Files.readAllLines(Paths.get(afterContextBeforePath));
        afterContextAfterLines = Files.readAllLines(Paths.get(afterContextAfterPath));

    }

    private void handleMissingFiles() {
        if (!new File(beforeContextPath).exists()) {
            beforeContextPath = beforeFilePath;
        }
        if (!new File(afterContextPath).exists()) {
            afterContextPath = beforeFilePath;
        }
        if (!new File(beforeNormalizedPath).exists()) {
            beforeNormalizedPath = beforeFilePath;
        }
        if (!new File(afterNormalizedPath).exists()) {
            afterNormalizedPath = beforeFilePath;
        }
        if (!new File(beforeContextBeforeNormalizedPath).exists()) {
            beforeContextBeforeNormalizedPath = beforeFilePath;
        }
        if (!new File(beforeContextAfterNormalizedPath).exists()) {
            beforeContextAfterNormalizedPath = beforeFilePath;
        }
        if (!new File(afterContextBeforeNormalizedPath).exists()) {
            afterContextBeforeNormalizedPath = beforeFilePath;
        }
        if (!new File(afterContextAfterNormalizedPath).exists()) {
            afterContextAfterNormalizedPath = beforeFilePath;
        }
        if (!new File(integratedChangePath).exists()) {
            integratedChangePath = beforeFilePath;
        }

    }

    private List<Pair<String, String>> getBeforeAfterListAux(List<String> before, List<String> after) {
        if (after.size() == before.size() - 1) { // in case the last line in the after file is empty
            after.add("");
        }
        for (int i = before.size(); i < beforeLines.size() ; i++) {
            before.add("");
            after.add("");
        }
        List<Pair<String, String>> beforeAfterList = new ArrayList<>();
        for (int i = 0; i < before.size(); i++) {
            beforeAfterList.add(new Pair<>(before.get(i), after.get(i)));
        }
        return beforeAfterList;
    }

    public List<Pair<String, String>> getBeforeAfterList() {
        return getBeforeAfterListAux(beforeLines, afterLines);
    }

    public List<Pair<String, String>> getBeforeContextBeforeAfterList() {
        return getBeforeAfterListAux(beforeContextBeforeLines, beforeContextAfterLines);
    }

    public List<Pair<String, String>> getAfterContextBeforeAfterList() {
        return getBeforeAfterListAux(afterContextBeforeLines, afterContextAfterLines);
    }

    public int dumpSamples(List<Triplet<Sample,Sample,Sample>> samples) throws IOException {
        List<String> sources = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue0)
                .map(s -> s.getSourceAsString(Common.PATH_SEP))
                .collect(Collectors.toList());
        List<String> targets = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue0)
                .map(s -> s.getTargetAsString(Common.PATH_SEP))
                .collect(Collectors.toList());
        List<String> editScripts = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue0)
                .map(s -> s.getEditScriptAsString(Common.PATH_SEP))
                .collect(Collectors.toList());
        List<String> idToken = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue0)
                .map(s -> s.getIdToeknAsString(Common.PATH_SEP))
                .collect(Collectors.toList());
        List<String> pathOps = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue0)
                .map(s -> s.getPathsOp(Common.PATH_SEP))
                .collect(Collectors.toList());
        List<String> dotTrees = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue0)
                .map(Sample::getDotString)
                .collect(Collectors.toList());
        List<String> beforeAST = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue0)
                .map(Sample::getASTbeforeString)
                .collect(Collectors.toList());
        List<String> afterAST = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue0)
                .map(Sample::getASTafterString)
                .collect(Collectors.toList());

        List<String> beforeContextPathes = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue1)
                .map(s -> s == null ? " " : s.getSourceAsString(Common.PATH_SEP))
                .collect(Collectors.toList());
        List<String> beforeContextTrees = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue1)
                .map(s -> s == null ? " " : s.getPathDotString())
                .collect(Collectors.toList());
        List<String> afterContextPathes = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue2)
                .map(s -> s == null ? " " : s.getSourceAsString(Common.PATH_SEP))
                .collect(Collectors.toList());
        List<String> afterContextTrees = samples.stream()
                .filter(Objects::nonNull)
                .map(Triplet::getValue2)
                .map(s -> s == null ? " " : s.getPathDotString())
                .collect(Collectors.toList());

        List<String> beforeContextLines = Files.readAllLines(Paths.get(beforeContextPath));
        List<String> afterContextLines = Files.readAllLines(Paths.get(afterContextPath));
        List<String> beforeNormalizedLines = Files.readAllLines(Paths.get(beforeNormalizedPath));
        List<String> afterNormalizedLines = Files.readAllLines(Paths.get(afterNormalizedPath));
        List<String> beforeContextBeforeLines = Files.readAllLines(Paths.get(beforeContextBeforePath));
        List<String> beforeContextAfterLines = Files.readAllLines(Paths.get(beforeContextAfterPath));
        List<String> afterContextBeforeLines = Files.readAllLines(Paths.get(afterContextBeforePath));
        List<String> afterContextAfterLines = Files.readAllLines(Paths.get(afterContextAfterPath));
        List<String> beforeContextBeforeNormalizedLines = Files.readAllLines(Paths.get(beforeContextBeforeNormalizedPath));
        List<String> beforeContextAfterNormalizedLines = Files.readAllLines(Paths.get(beforeContextAfterNormalizedPath));
        List<String> afterContextBeforeNormalizedLines = Files.readAllLines(Paths.get(afterContextBeforeNormalizedPath));
        List<String> afterContextAfterNormalizedLines = Files.readAllLines(Paths.get(afterContextAfterNormalizedPath));
        List<String> integratedChangeLines = Files.readAllLines(Paths.get(integratedChangePath));

        List<String> beforeFiltered = new ArrayList<>();
        List<String> afterFiltered = new ArrayList<>();
        List<String> beforeNormalizedFiltered = new ArrayList<>();
        List<String> afterNormalizedFiltered = new ArrayList<>();
        List<String> beforeContextFiltered = new ArrayList<>();
        List<String> afterContextFiltered = new ArrayList<>();
        List<String> beforeContextBeforeFiltered = new ArrayList<>();
        List<String> afterContextBeforeFiltered = new ArrayList<>();
        List<String> beforeContextAfterFiltered = new ArrayList<>();
        List<String> afterContextAfterFiltered = new ArrayList<>();
        List<String> beforeContextBeforeNormalizedFiltered = new ArrayList<>();
        List<String> afterContextBeforeNormalizedFiltered = new ArrayList<>();
        List<String> beforeContextAfterNormalizedFiltered = new ArrayList<>();
        List<String> afterContextAfterNormalizedFiltered = new ArrayList<>();
        List<String> integratedChangeFiltered = new ArrayList<>();
        int i = 1;
        for (Triplet<Sample,Sample,Sample> s : samples) {
            if (s != null) {
                beforeFiltered.add(beforeLines.get(i-1));
                afterFiltered.add(afterLines.get(i-1));
                beforeNormalizedFiltered.add(beforeNormalizedLines.get(i-1));
                afterNormalizedFiltered.add(afterNormalizedLines.get(i-1));
                beforeContextFiltered.add(beforeContextLines.get(i-1));
                afterContextFiltered.add(afterContextLines.get(i-1));
                beforeContextBeforeFiltered.add(beforeContextBeforeLines.get(i-1));
                beforeContextAfterFiltered.add(beforeContextAfterLines.get(i-1));
                afterContextBeforeFiltered.add(afterContextBeforeLines.get(i-1));
                afterContextAfterFiltered.add(afterContextAfterLines.get(i-1));
                beforeContextBeforeNormalizedFiltered.add(beforeContextBeforeNormalizedLines.get(i-1));
                beforeContextAfterNormalizedFiltered.add(beforeContextAfterNormalizedLines.get(i-1));
                afterContextBeforeNormalizedFiltered.add(afterContextBeforeNormalizedLines.get(i-1));
                afterContextAfterNormalizedFiltered.add(afterContextAfterNormalizedLines.get(i-1));
                integratedChangeFiltered.add(integratedChangeLines.get(i-1));
            }
            i++;
        }

        Files.write(Paths.get(projectDir + projectName + ".edit_script"), editScripts, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".path"), sources, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_ctx_path"), beforeContextPathes, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_ctx_path"), afterContextPathes, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".path_op"), pathOps, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".ids"), idToken, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".label"), targets, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".dot_trees"), dotTrees, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_ast_trees"), beforeAST, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_ast_trees"), afterAST, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_ctx_dot_trees"), beforeContextTrees, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_ctx_dot_trees"), afterContextTrees, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_filtered"), beforeFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_filtered"), afterFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_normalized_filtered"), beforeNormalizedFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_normalized_filtered"), afterNormalizedFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_ctx_filtered"), beforeContextFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_ctx_filtered"), afterContextFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_ctx_before_filtered"), beforeContextBeforeFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_ctx_after_filtered"), beforeContextAfterFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_ctx_before_filtered"), afterContextBeforeFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_ctx_after_filtered"), afterContextAfterFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_ctx_before_normalized_filtered"), beforeContextBeforeNormalizedFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".before_ctx_after_normalized_filtered"), beforeContextAfterNormalizedFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_ctx_before_normalized_filtered"), afterContextBeforeNormalizedFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".after_ctx_after_normalized_filtered"), afterContextAfterNormalizedFiltered, Charset.defaultCharset());
        Files.write(Paths.get(projectDir + projectName + ".integrated_change_filtered"), integratedChangeFiltered, Charset.defaultCharset());
        return targets.size();
    }
}
