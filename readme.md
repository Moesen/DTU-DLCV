```
____     ______  __  __                   ____     __       ____     __  __    
/\  _`\  /\__  _\/\ \/\ \                 /\  _`\  /\ \     /\  _`\  /\ \/\ \   
\ \ \/\ \\/_/\ \/\ \ \ \ \                \ \ \/\ \\ \ \    \ \ \/\_\\ \ \ \ \  
 \ \ \ \ \  \ \ \ \ \ \ \ \      _______   \ \ \ \ \\ \ \  __\ \ \/_/_\ \ \ \ \ 
  \ \ \_\ \  \ \ \ \ \ \_\ \    /\______\   \ \ \_\ \\ \ \L\ \\ \ \L\ \\ \ \_/ \
   \ \____/   \ \_\ \ \_____\   \/______/    \ \____/ \ \____/ \ \____/ \ `\___/
    \/___/     \/_/  \/_____/                 \/___/   \/___/   \/___/   `\/__/ 
```

## Notes
### **Link to overview of optuna and wandb integration for creating cool sweep plots**
[https://www.h4pz.co/blog/2020/10/3/optuna-and-wandb](Link)

## **Note about wandb**
To make wandb work without logging in, set api key in environment.
Key should have name:
`WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxx`

### *Make src package imports work*
Encountered quite a problem trying to do `import src.module`.
Best solution I found was to create a package from it and install it in my env.
The package specifics is placed in the `setup.py` file, and to install run

- `pip install -e .`
- `conda develop .`

Both are listed, but its proberbly best only to use venv or conda, so choose one ;)

### **TODO**
- [x] Add makes for root makefile to make things faster
