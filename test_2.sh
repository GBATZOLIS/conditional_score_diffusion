ACTIVATE="tmux new-window;
          tmux send \"zsh\" ENTER; 
          sleep 10; 
          tmux send \"conda activate score_sde\"  ENTER; 
          sleep 10;"
tmux kill-ses -t test
tmux new-session -d -s test
$ACTIVATE
