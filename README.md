
**Installation**
--------------

1. Clone the repository using 

```bash
    git clone https://github.com/doulatdutta/ai-algo-trading-framework.git
```

2. Rename `example.env` file in the config folder as `.env`

3.  Set up a virtual environment (optional but recommended):

```bash
   python -m venv venv
```

4. Activate virtual environment

```bash
   source venv/bin/activate
  ```

* On Windows use 

```bash
   venv\Scripts\activate
``` 

* On Windows use If you encounter an error regarding execution policies, you can temporarily bypass it by running:
```bash
   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
   ```
5. Install the dependencies using 

```bash
    pip install -r requirements.txt
```

6. test run for manager agent

```bash
    python -m agents.manager_agent.manager_agent
```
7. test run for algo_agent

```bash
    python -m agents.algo_agent.algo_agent
```

8. test run for data_downloaded

```bash
    python -m agents.backtesting_agent.data_download
```

9. test run for backtesting_agent

```bash
    python -m agents.backtesting_agent.backtesting_agent
```

10. test run for checker_agent

```bash
    python -m agents.checker_agent.checker_agent
```

11. test run for automated_feedback

```bash
    python -m agents.automated_feedback.automated_feedback
```





