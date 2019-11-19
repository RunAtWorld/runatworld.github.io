# git push 本地不用输入用户名密码

1. 进入~（用户）目录: `cd ~`
2. 建立文件 .git-credentials: `touch  .git-credentials`

3. 编辑文件 .git-credentials： `vi  .git-credentials`

4. 添加 https://用户名:密码@gitlab.com

5. 执行命令：`git config --global credential.helper store`

6. 查看文件：`more .gitconfig` 可以看到如下信息，设置成功。

    ```
    [push]
        default = simple
    [user]
        name = test
        email = test@gitlab.com
    ```

4，5步骤也可以换成，在 `.git-credentials` 中加上如下内容

```
[user]
    email = test@gitlab.com
    name = test

[credential]
    helper = store
```
