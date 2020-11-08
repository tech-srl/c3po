package Extractor;

import Extractor.Model.*;
import Extractor.Utils.CommandLineValues;
import com.github.gumtreediff.client.Run;
import org.javatuples.Pair;
import org.javatuples.Triplet;
import org.kohsuke.args4j.CmdLineException;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;


public class App {
    private static CommandLineValues commandLineValues;
    private static boolean recordChildren;
    private static String legalChildrenMapPath = "legalChildrenMap.ser";

    private static Pair<Long,Long> projectProcessTask(String projectName)  {
        long numOfSamples = 0;
        long totalSamples = 0;
        try {
        String projectDir = commandLineValues.projectsDir + File.separator + projectName + File.separator;
        if (!new File(projectDir).isDirectory()) {
            return null;
        }
        Project project = new Project(projectDir, projectName);
        List<Pair<String, String>> beforeAfterList = project.getBeforeAfterList();
        List<Pair<String, String>> beforeContextBeforeAfterList = project.getBeforeContextBeforeAfterList();
        List<Pair<String, String>> afterContextBeforeAfterList = project.getAfterContextBeforeAfterList();
        totalSamples = beforeAfterList.size();
        ExecutorTask.initialCounters(beforeAfterList.size(), projectName);

        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(commandLineValues.numThreads);
        List<ExecutorTask> tasks = new ArrayList<>();
        for(int i = 0; i < beforeAfterList.size(); i++) {
            tasks.add(new ExecutorTask(commandLineValues, beforeAfterList.get(i), beforeContextBeforeAfterList.get(i), afterContextBeforeAfterList.get(i)));
        }
        List<Future<Triplet<Sample,Sample,Sample>>> futureList = new ArrayList<>();
        try {
            futureList = executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
        List<Triplet<Sample,Sample,Sample>> samples = futureList.stream()
                                         .map(f -> {
                                                        try {
                                                            return f.get();
                                                        } catch (Exception e) {
                                                            e.printStackTrace();
                                                            return null;
                                                       }
                                             })
                                         .collect(Collectors.toList());
        numOfSamples = project.dumpSamples(samples);
        } catch (Exception e) {
            System.out.println(" " + projectName);
            e.printStackTrace();
        }
        if (recordChildren) {
            System.out.println(projectName);
        } else {
            System.out.println();
        }
        return new Pair<>(numOfSamples, totalSamples);
    }

    public static void main(String[] args) {

        Instant start = Instant.now();
        try {
            commandLineValues = new CommandLineValues(args);
        } catch (CmdLineException e) {
            e.printStackTrace();
            return;
        }
        ExecutorTask.commandLineValues = commandLineValues;
        Run.initGenerators();
        File projectsDir = new File(commandLineValues.projectsDir);
        File legalChildrenMapFile = new File(commandLineValues.projectsDir + "/" + legalChildrenMapPath);
        if (!legalChildrenMapFile.exists()) {
            System.out.println("++++++++++++++++++++++++++++++++");
            System.out.println("Only recording children legal map!");
            System.out.println("++++++++++++++++++++++++++++++++");
            Node.legalChildMap = new HashMap<>();
            ExecutorTask.recordChildren = true;
        } else {
            ExecutorTask.recordChildren = false;
            try
            {
                FileInputStream fis = new FileInputStream(legalChildrenMapFile.getPath());
                ObjectInputStream ois = new ObjectInputStream(fis);
                Node.legalChildMap = (HashMap) ois.readObject();
                ois.close();
                fis.close();

            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }

        ExecutorTask.process(new Pair<>(
                        "var item = nodes.Where(x => x.Name == n && x.Type == NodeType.Directory && x.ParentId == parent.Id).FirstOrDefault();",
                        "var item = nodes.FirstOrDefault(x => x.Name == n && x.Type == NodeType.Directory && x.ParentId == parent.Id);"),
                true);

        System.out.println("Processing...");
        String[] projectsArray = projectsDir.list();
        long samplesCount = 0;
        long totalCount = 0;
        if (projectsArray != null) {
            List<String> projectsPaths = Arrays.asList(projectsArray);
            List<Pair<Long,Long>> results = projectsPaths.stream()
                    .map(App::projectProcessTask)
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());
            samplesCount = results.stream()
                    .map(Pair::getValue0)
                    .mapToLong(Long::valueOf).sum();
            totalCount = results.stream()
                    .map(Pair::getValue1)
                    .mapToLong(Long::valueOf).sum();
        }
        if (!legalChildrenMapFile.exists()) {
            try
            {
                FileOutputStream fos = new FileOutputStream(legalChildrenMapFile.getPath());
                ObjectOutputStream oos = new ObjectOutputStream(fos);
                oos.writeObject(Node.legalChildMap);
                oos.close();
                fos.close();
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
            System.out.println("Serialized legal children map in " + legalChildrenMapFile.getPath());
        }

        Instant finish = Instant.now();
        long timeElapsed = Duration.between(start, finish).toMinutes();
        System.out.println("\nCreated " + samplesCount + " samples out of " + totalCount + "in " + timeElapsed + "minutes");
        System.out.println("\n MOV: " + ExecutorTask.movActions.get() + " subTreeSize: " + ExecutorTask.movSubTreeSizes.get() + " INS: " + ExecutorTask.insActions.get() + " subTreeSize: " + ExecutorTask.insSubTreeSizes.get()
                + " DEL: " + ExecutorTask.delActions.get() + " subTreeSize: " + ExecutorTask.delSubTreeSizes.get());

        //        System.out.println("\n MOV: " + ExecutorTask.movActions.get() + "DEL: " + ExecutorTask.delActions.get() + "UPD: " + ExecutorTask.updActions.get() + "INS: " + ExecutorTask.insActions.get());
        System.out.println("Done!");
    }
}
