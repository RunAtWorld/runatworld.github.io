# git常用操作
 
git help config	查看git配置帮助

git config --global user.name "lipengfei"	配置用户名   
git config --global user.email "goodlpf00@gmail.com"	配置电子邮箱   
ssh-keygen -t rsa -C "goodlpf00@gmail.com"	生成公钥和私钥对  
ssh git@bitbucket.org	测试ssh的密钥对  
//在linux下，如果测试不通过，键入命令: add-ssh ~/.ssh/id_rsa   

git clone git@github.com:RunAtWorld/quietalk.git	从远程克隆一个仓库   

git add 1.txt  添加1.txt到下一次提交,并被跟踪   
git add .  添加当前文件夹下所有文件到下一次提交,并被跟踪   
git add *.txt 添加所有txt文件到下一次提交,并被跟踪   
git commit 提交前一次变化到本地仓库   
git commit -m "abc"	带信息提交一次变化到本地仓库   
git commit -am "abc"	带信息提交所有修改的文件变化到本地仓库   
git commit --amend	修改上一次提交   
git status 显示当前git的状态   
git log	查看git日志     

git branch	查看当前分支   
git branch {new-branch}	创建新的分支   
git checkout {branch1}	切换到一个分支branch1  
git checkout -b {branch3}	创建并切换到分支branch3   

git merge {branch4}	当前分支合并至branch4分支的HEAD指针处  
git branch -d {branch4}	删除一个本地分支branch4  

git stash 储藏当前状态，切换到其他分支  
git stash list	查看储藏状态的列表  
git stash apply  {stash-name}	回到原来的分支，恢复之前的工作状态  

git push {remote} {branch}	推送本地分支到远程仓库  
git push -u {remote} {branch}	推送本地分支到远程仓库  
git push -u {remote} --all	推送本地仓库所有数据到到远程仓库,-u表示建立跟踪  

git log --no-merges {remote}/{branch}	列出远程没有合并前的变化  
git fetch {remote}	从远程下载所有变化数据,但是不放到工作区  
git merge {remote}/{branch}	将本地仓库版本合并至远程仓库  
git pull {remote} {branch}	相当于git fetch {remote}和git merge {remote}/{branch}两条命令的合并，自动抓取数据并将本地仓库版本合并至远程仓库

git remote -v	列出当前配置的远程库  
git branch -r	列出本地分支与远程分支的对应关系  
git branch -v	列出远程分支的版本操作情况  
git checkout --track {local-branch} {remote}/{branch}	新建一个本地分支并与远程分支关联  
git branch --set-upstream {local-branch} {remote}/{remote-branch}	将本地分支local_branch与远程的remote_branch建立联系  

git remote show {remote}	列出远程仓库的信息  
git remote	rename {old-name} {new-name}	重命名远程仓库  
git remote rm {remote}	删除远程仓库remote  

git reset --hard	重置仓库    
![Git各个状态之间转换指令总结](./gitcmd_files/clip_image001.png)

Git各个状态之间转换指令总结
基本状态标识 

* A- = untracked 未跟踪
* A = tracked 已跟踪未修改
* A+ = modified 已修改未暂存
* B = staged 已暂存未提交
* C = committed 已提交未PUSH
 
各状态之间变化

* A- -> B : git add {file}  
* B -> A- : git rm --cached {file}   
* B -> 删除不保留文件 : git rm -f {file}    
* A -> A- : git rm --cached {file}  
* A -> A+ : 修改文件  
* A+ -> A : git checkout -- {file}  
* A+ -> B : git add {file}  
* B -> A+ : git reset HEAD {file}  
* B -> C : git commit  
* C -> B : git reset --soft HEAD   
* 修改最后一次提交: git commit –amend  

# FAQ
1.SSH生成id_rsa,id_rsa.pub后，连接服务器却报：Agent admitted failure to sign using the key错误。  
解决：
在当前用户下执行命令：
ssh-add
即可解决。
 
 
 
 