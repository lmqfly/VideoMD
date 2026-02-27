# 将 DiffSynth-Studio 推送到 GitHub 的流程

## 一、本地已完成的准备

- 项目已是 Git 仓库（`git init` 已执行）。
- `.gitignore` 已配置，会忽略：
  - **video_data/**（视频数据目录）
  - **models/**（模型目录）
  - 以及常见 Python/日志/临时文件。

## 二、在 GitHub 上创建仓库

1. 登录 [GitHub](https://github.com)。
2. 点击右上角 **“+”** → **“New repository”**。
3. 填写：
   - **Repository name**：例如 `DiffSynth-Studio`
   - **Description**：可选
   - **Public** 或 **Private** 按需选择
   - **不要**勾选 “Add a README file”等（本地已有代码）。
4. 点击 **“Create repository”**。
5. 记下仓库地址，例如：`https://github.com/你的用户名/DiffSynth-Studio.git` 或 `git@github.com:你的用户名/DiffSynth-Studio.git`。

## 三、本地执行命令（按顺序运行）

在项目根目录下打开终端，执行：

```bash
# 1. 进入项目目录
cd /home/xujunzhang/mingquan/DiffSynth-Studio-main

# 2. 确认 .gitignore 已忽略 video_data 和 models（可选检查）
cat .gitignore | grep -E "video_data|models"

# 3. 添加要提交的文件（video_data/ 和 models/ 会被自动忽略）
git add .

# 4. 查看将要提交的文件（确认没有 video_data、models）
git status

# 5. 首次提交
git commit -m "Initial commit: DiffSynth-Studio"

# 6. 添加远程仓库（把下面的 URL 换成你在 GitHub 上创建的仓库地址）
git remote add origin https://github.com/你的用户名/DiffSynth-Studio.git
# 若使用 SSH：
# git remote add origin git@github.com:你的用户名/DiffSynth-Studio.git

# 7. 推送到 GitHub（首次推送并设置上游分支）
git branch -M main
git push -u origin main
```

如果 GitHub 上创建的是空仓库且默认分支叫 `master`，可以用：

```bash
git push -u origin master
```

## 四、若已存在 origin 或要更换仓库地址

```bash
# 查看当前远程
git remote -v

# 删除原有 origin（如需更换仓库）
git remote remove origin

# 重新添加并推送
git remote add origin https://github.com/你的用户名/新仓库名.git


cd ~/mingquan/DiffSynth-Studio-main
git config --local -l | egrep -i 'remote\.origin\.url|https?\.proxy|http\.version'
git remote set-url origin https://github.com/lmqfly/VideoMD.git
git config --local http.proxy  http://booster.internal.puhui.chengfengerlai.com:30090
git config --local https.proxy http://booster.internal.puhui.chengfengerlai.com:30090
git config --local http.version HTTP/1.1
git push -u origin main
lmqfly
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

```

## 五、之后日常更新推送

```bash
cd /home/xujunzhang/mingquan/DiffSynth-Studio-main
git add .
git commit -m "描述你的修改"
git push
```

## 六、确认 video_data 和 models 未被提交

- 推送前用 `git status` 确认列表里没有 `video_data/`、`models/`。
- 在 GitHub 仓库页面的 “Code” 里查看，不应出现这两个目录。

若之前误提交过这两个目录，需要从 Git 历史中移除（保留本地文件）：

```bash
git rm -r --cached video_data/ models/
git commit -m "Stop tracking video_data and models"
git push
```

---

**总结**：已忽略 `video_data/` 和 `models/`，按上述流程在 GitHub 建仓并执行第三部分的命令即可完成首次推送；日常更新用第五部分即可。
