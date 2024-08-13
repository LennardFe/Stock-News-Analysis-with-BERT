# Setup

<div align="justify">
<p>Step-by-step instructions for setting up and configuring the project for a successful start. The following steps are mostly designed to work for Windows 10 and 11, where they have also been tested. For Mac and any Linux Distribution, please refer to external sources.</p>
</div>



### Installing python
<div align="justify">
<p>Install Python <b>3.11.5</b>, ensuring compatibility with the project requirements. For a windows and mac installer, refer to the <a href="https://www.python.org/downloads/release/python-3115/">python website</a>, scroll down to files and choose the fitting installer. To install Python on Linux, you can use a package manager like <i>apt</i> and you may need external repositories or version managers to get the older 3.11.5 version.</p>
</div>

### Downloading the project
<div align="justify">
<p>Start by navigating to the GitHub page of this project. Locate the green <i>Code</i> button on the top right and click it. You can either clone it using the command prompt with <i>git clone</i> followed by the shown URL, download the zip file, or open it directly with GitHub Desktop. After that navigate to the project folder and continue with the next step.</p>
</div>

### Creating a Virtual Enviroment (Recommended)
<div align="justify">
<p>After setting up Python and having downloaded the project, I would recommend setting up a virtual environment to isolate the project and its packages, to avoid interference with a possible global Python installation. To do that, you can use the following command on any system:</p>
</div>

```
python -m venv .venv
```

<div align="justify">
<p>This will create a new folder called <i>.venv</i> containing the virtual environment. Run the next command to activate the created environment. After running this, you should see the virtual environment's name (.venv) show up in the command prompt.</p>
</div>

```
.venv\Scripts\Activate
```

<div align="justify">
<p>While testing this, I was facing the problem of not being able to execute any scripts on my system. To get this working, I had to start <b>PowerShell as an admin</b> and execute the following command:</p>
</div>

```
Set-ExecutionPolicy Unrestricted -Force
```

<div align="justify">
<p>This policy allows to execute scripts without any restrictions, which can be useful and even necessary, as in this example, to execute our script to activate the virtual environment, but also potentially risky if the scripts are untrusted. I would recommend undoing this change after using the relevant script, with the following command:</p>
</div>

```
Set-ExecutionPolicy Restricted
```

<div align="justify">
<p>This sets the execution policy back to the Windows default, where no scripts can be executed.</p>
</div>

### Adding API-Key
<div align="justify">
<p>As mentioned earlier the project depends an the Leeway-API, so for all the data we retrieve we need a valid API-Key. To get one, sign up on <a href="https://leeway.tech/">leeway.tech</a> and take a look at the Data-API Package. After receiving your API-Key, create a folder <i>assets</i>, with a file named <i>credential.py</i> inside, containing the following line:</p>
</div>

```
apikey="your_api_key_here"
```

###  Install packages
<div align="justify">
<p>As the final step, we can execute our main script <i>executor.py</i> to start the program. Before the notebook gets executed, a function will be called to install all packages listed in the <i>requirements.txt</i>. Even though we don't really need all of them, with all these dependencies, I still include every single one to ensure there are no compatibility problems. After the installation, the program will be executed with the set parameters from the file. For more information about them, refer to the <a href="https://github.com/LennardFe/Stock-News-Analysis-with-BERT/blob/main/docs/methodology.md">methodology</a> or the docstrings in the notebook itself. Following the initial execution, you can remove the function call to check for packages, comment it out, or change the <i>already_checked</i> parameter to <i>True</i>. This will prevent the code from checking for the packages every time you start it.</p>
</div>
