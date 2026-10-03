using System;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Threading;
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
                launcherScript = Path.Combine(baseDir, "mem_v3", "application", "control_center.py");
            }

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
            ProcessStartInfo pyInfo = new ProcessStartInfo();
            pyInfo.FileName = pythonExe;
            pyInfo.Arguments = "\"" + launcherScript + "\"";
            pyInfo.WorkingDirectory = baseDir;
            pyInfo.CreateNoWindow = true;
            pyInfo.UseShellExecute = false;
            pyInfo.WindowStyle = ProcessWindowStyle.Hidden;

            Process pyProcess = null;
            try
            {
                pyProcess = Process.Start(pyInfo);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Falha ao iniciar o servico Python:\n" + ex.Message,
                    "MEM LLM Orchestrator - Erro",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            // 4. Wait for server on port 8089 to become responsive
            int port = 8089;
            bool serverReady = WaitForPort(port, 15); // up to 15 seconds

            // 5. Open Desktop App Window
            string appUrl = "http://127.0.0.1:" + port;
            string browserExe = FindBrowser();

            Process browserProcess = null;
            if (!string.IsNullOrEmpty(browserExe) && File.Exists(browserExe))
            {
                ProcessStartInfo brInfo = new ProcessStartInfo();
                brInfo.FileName = browserExe;
                brInfo.Arguments = "--app=" + appUrl + " --window-size=1366,860 --disable-extensions";
                brInfo.UseShellExecute = false;
                browserProcess = Process.Start(brInfo);
            }
            else
            {
                Process.Start(appUrl);
            }

            // If browser window was started directly, wait for user to close it
            if (browserProcess != null)
            {
                try
                {
                    browserProcess.WaitForExit();
                }
                catch
                {
                }
            }

            // 6. Cleanup Python Process on exit
            if (pyProcess != null && !pyProcess.HasExited)
            {
                try
                {
                    pyProcess.Kill();
                }
                catch
                {
                }
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

            // Check PATH environment variable
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

        static string FindBrowser()
        {
            string[] candidates = new string[]
            {
                @"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                @"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                @"C:\Program Files\Google\Chrome\Application\chrome.exe",
                @"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Microsoft\Edge\Application\msedge.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Google\Chrome\Application\chrome.exe")
            };

            foreach (string p in candidates)
            {
                if (File.Exists(p)) return p;
            }

            return null;
        }

        static bool WaitForPort(int port, int timeoutSeconds)
        {
            DateTime deadline = DateTime.Now.AddSeconds(timeoutSeconds);
            while (DateTime.Now < deadline)
            {
                try
                {
                    using (TcpClient client = new TcpClient())
                    {
                        IAsyncResult res = client.BeginConnect("127.0.0.1", port, null, null);
                        bool success = res.AsyncWaitHandle.WaitOne(500);
                        if (success && client.Connected)
                        {
                            client.EndConnect(res);
                            return true;
                        }
                    }
                }
                catch
                {
                }
                Thread.Sleep(300);
            }
            return false;
        }
    }
}
