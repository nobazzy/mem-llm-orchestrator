using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace MemOrchestrator
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            // 1. Locate Python Interpreter
            string pythonExe = FindPython(baseDir);
            if (string.IsNullOrEmpty(pythonExe) || !File.Exists(pythonExe))
            {
                MessageBox.Show(
                    "O ambiente virtual Python nao foi encontrado.\n\n" +
                    "Certifique-se de que a pasta 'mem_v3\\.venv' existe com o Python configurado.",
                    "MEM LLM Orchestrator - Erro de Inicializacao",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            // 2. Locate Launcher script
            string launcherScript = Path.Combine(baseDir, "MEM_Launcher.py");
            if (!File.Exists(launcherScript))
            {
                MessageBox.Show(
                    "O script inicializador 'MEM_Launcher.py' nao foi localizado.",
                    "MEM LLM Orchestrator - Erro",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            // 3. Start Python Background Process
            // MEM_Launcher.py handles starting the HTTP server, verifying connection,
            // launching the isolated desktop browser window, and waiting for user close.
            ProcessStartInfo pyInfo = new ProcessStartInfo();
            pyInfo.FileName = pythonExe;
            pyInfo.Arguments = "\"" + launcherScript + "\"";
            pyInfo.WorkingDirectory = baseDir;
            pyInfo.CreateNoWindow = true;
            pyInfo.UseShellExecute = false;
            pyInfo.WindowStyle = ProcessWindowStyle.Hidden;

            try
            {
                using (Process pyProcess = Process.Start(pyInfo))
                {
                    if (pyProcess != null)
                    {
                        pyProcess.WaitForExit();
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Falha ao executar o inicializador:\n" + ex.Message,
                    "MEM LLM Orchestrator - Erro",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }

        static string FindPython(string baseDir)
        {
            string[] candidates = new string[]
            {
                Path.Combine(baseDir, "mem_v3", ".venv", "Scripts", "python.exe"),
                Path.Combine(baseDir, ".venv", "Scripts", "python.exe"),
                Path.Combine(baseDir, "venv", "Scripts", "python.exe"),
                Path.Combine(baseDir, "python.exe")
            };

            foreach (string p in candidates)
            {
                if (File.Exists(p)) return p;
            }

            string pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathEnv))
            {
                string[] paths = pathEnv.Split(';');
                foreach (string p in paths)
                {
                    string candidate = Path.Combine(p.Trim(), "python.exe");
                    if (File.Exists(candidate)) return candidate;
                }
            }

            return null;
        }
    }
}
