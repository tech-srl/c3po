package Extractor.Utils;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

/**
 * This class handles the programs arguments.
 */
public class CommandLineValues {
	@Option(name = "--projects_dir", required = true)
	public String projectsDir = null;
	@Option(name = "--max_nodes", required = false)
	public int maxNodes = 20;
	@Option(name = "--no_INS", required = false)
	public boolean noINS = false;
	@Option(name = "--num_threads", required = false)
	public int numThreads = 64;
	@Option(name = "--max_path_ctx", required = false)
	public int maxPathCtx = 20;
	@Option(name = "--create_inference_sample", required = false)
	public boolean inferenceSample = false;

	public CommandLineValues(String... args) throws CmdLineException {
		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			System.err.println(e.getMessage());
			parser.printUsage(System.err);
			throw e;
		}
	}

	public CommandLineValues() {

	}
}