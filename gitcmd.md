# Git常用操作
git help config	查看git配置帮助

### 配置账户
```
git config --global user.name "lipengfei"	配置用户名   
git config --global user.email "goodlpf00@gmail.com"	配置电子邮箱   
ssh-keygen -t rsa -C "goodlpf00@gmail.com"	生成公钥和私钥对  
ssh git@bitbucket.org	测试ssh的密钥对  #在linux下，如果测试不通过，键入命令: add-ssh ~/.ssh/id_rsa   
```

### 克隆仓库
```
git clone git@github.com:RunAtWorld/quietalk.git	从远程克隆一个仓库   
```

### 提交文件
```
git add 1.txt  添加1.txt到下一次提交,并被跟踪   
git add .  添加当前文件夹下所有文件到下一次提交,并被跟踪   
git add *.txt 添加所有txt文件到下一次提交,并被跟踪   
git commit 提交前一次变化到本地仓库   
git commit -m "abc"	带信息提交一次变化到本地仓库   
git commit -am "abc"	带信息提交所有修改的文件变化到本地仓库   
git commit --amend	修改上一次提交   
git status 显示当前git的状态   
git log	查看git日志     
```

### 分支
1. 创建分支
```
git branch	#查看当前分支   
git branch dev	#创建dev分支   
git checkout dev	#切换到一个分支dev 
git checkout -b dev	#创建并切换到分支dev
```

1. 删除分支
```
git branch -d {branch4}	删除一个本地分支branch4  
```

1. 合并分支
```
git merge {branch4} 当前分支合并至branch4分支的HEAD指针处 
```

1. 查看分支
```
git branch -r   列出远程分支
git branch -v   列出本地分支
git branch -a   列出本地和远程所有分支
```

1. 与远程分支建立联系
```
git checkout --track dev_loacal origin/dev   新建一个本地dev_loacal分支并与远程dev分支关联  
git branch --set-upstream-to dev_loacal origin/dev  将本地分支dev_loacal与远程的dev建立联系  
```

1. 储存当前状态
```
git stash #储藏当前状态，切换到其他分支  
git stash list	#查看储藏状态的列表  
git stash apply  {stash_name}	#回到原来某个工作状态，恢复之前的工作状态  
```

git push {远程主机名} {本地分支名}:{远程分支名}	 #推送本地分支到远程仓库  

git log --no-merges {remote}/{branch}	列出远程没有合并前的变化  
git fetch {remote}	从远程下载所有变化数据,但是不放到工作区  
git merge {remote}/{branch}	将本地仓库版本合并至远程仓库  
git pull {remote} {branch}	相当于git fetch {remote}和git merge {remote}/{branch}两条命令的合并，自动抓取数据并将本地仓库版本合并至远程仓库

git remote -v	列出当前配置的远程库  


git remote show {remote}	列出远程仓库的信息  
git remote	rename {old-name} {new-name}	重命名远程仓库  
git remote rm {remote}	删除远程仓库remote  

git reset --hard	重置仓库    

# Git各个状态之间转换指令总结   
![Git各个状态之间转换指令总结](./gitcmd_files/1352126739_7909.jpg) <br>  
**工作区**：就是你在电脑里能看到的目录。<br>
**暂存区**：英文叫stage, 或index。一般存放在 ".git目录下" 下的index文件（.git/index）中，所以我们把暂存区有时也叫作索引（index）。<br>
**版本库**：工作区有一个隐藏目录.git，这个不算工作区，而是Git的版本库。<br>
<br>
在版本库中标记为 "index" 区域是暂存区（stage, index），标记为 "master" 的是 master 分支所代表的目录树。<br>
"HEAD" 实际是指向 master 分支的一个"游标"。图中出现 HEAD 的地方可以用 master 来替换。objects 标识的区域为 Git 的对象库，实际位于 ".git/objects" 目录下，里面包含了创建的各种对象及内容。<br>

- 当对工作区修改（或新增）的文件执行 `git add` 命令时，暂存区的目录树被更新，同时工作区修改（或新增）的文件内容被写入到对象库中的一个新的对象中，而该对象的ID被记录在暂存区的文件索引中。<br>
- 当执行提交操作 `git commit` 时，暂存区的目录树写到版本库（对象库）中，master 分支会做相应的更新。即 master 指向的目录树就是提交时暂存区的目录树。<br>
- 当执行 `git reset HEAD` 命令时，暂存区的目录树会被重写，被 master 分支指向的目录树所替换，但是工作区不受影响。<br>
- 当执行 `git rm --cached <file>` 命令时，会直接从暂存区删除文件，工作区则不做出改变。<br>
- 当执行 `git checkout .` 或者 `git checkout -- <file>` 命令时，会用暂存区全部或指定的文件替换工作区的文件。这个操作很危险，会清除工作区中未添加到暂存区的改动。<br>
- 当执行 `git checkout HEAD .` 或者 `git checkout HEAD <file>` 命令时，会用 HEAD 指向的 master 分支中的全部或者部分文件替换暂存区和以及工作区中的文件。这个命令也是极具危险性的，因为不但会清除工作区中未提交的改动，也会清除暂存区中未提交的改动。<br>

# FAQ
1. SSH生成id_rsa,id_rsa.pub后，连接服务器却报：Agent admitted failure to sign using the key错误。  
解决：
在当前用户下执行命令：
ssh-add
即可解决。

1. 本地文件夹与远程库建立关联
```
git init .
git remote add origin git@github.com:RunAtWorld/ceph_manual.git  #与远程主机关联
git branch --set-upstream-to=origin/master master #建立远程master分支与本地master分支的关联
```
 
# 参考
1. [易百git教程：https://www.yiibai.com/git/git_basic_concepts.html](https://www.yiibai.com/git/git_basic_concepts.html)
2. [git官方教程：https://git-scm.com/book/zh/v2/](https://git-scm.com/book/zh/v2/)
3. [git菜鸟教程:http://www.runoob.com/git/git-tutorial.html](http://www.runoob.com/git/git-tutorial.html)
4. [git简明指南:http://rogerdudler.github.io/git-guide/index.zh.html](http://rogerdudler.github.io/git-guide/index.zh.html)