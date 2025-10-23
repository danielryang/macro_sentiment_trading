@echo off
echo Merging all feature branches...

git merge feat/signal-generation --no-ff -m "merge: integrate signal generation system"
git merge feat/cli-interface --no-ff -m "merge: integrate CLI interface"
git merge feat/notebooks --no-ff -m "merge: integrate Jupyter notebooks"
git merge feat/testing --no-ff -m "merge: integrate testing suite"

echo All branches merged successfully!
git branch -a

