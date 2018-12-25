
# Title: 
Monte Carlo simulation: Predicting NBA western conference champion of season 2017-2018

## Team Member(s):
Xiang Chen, Mengyuan Li, Yushuo Fan, Ruoqiao Zhang(Thursday)

# Monte Carlo Simulation Scenario & Purpose:
Purpose: This program aims to use Monte Carlo Methods to predict NBA western conference champion of season 2017-2018. After the execution of Monte Carlo simulation of the match, the team who has the highest probabilty of winning (calulated by predicted winning times / simulation times) will be the predicted winner.</br>

Scenario: The match is going on now and has come to the playoff. There are 8 teams surviving to compete for the champion.These 8 teams are ranked according to their performance in previous matches and we represented them as
Team1, Team2,...,Team8.</br>

**The 1st turn:**</br>     Team1 VS Team8 ===> Winner(1,8)</br>
                  Team2 VS Team7 ===> Winner(2,7)</br>
                  Team3 VS Team6 ===> Winner(3,6)</br>
                  Team4 VS Team5 ===> Winner(4,5)</br>

**The 2nd turn:**</br>     Winner(1,8) VS Winner(4,5) ===> Final_Team1</br>
                  Winner(2,7) VS Winner(3,6) ===> Final_Team2</br>

**The 3rd turn:**</br>     Final_Team1 VS Final_Team2 ===> Champion</br>
          

## Simulation's variables of uncertainty

There are 3 random varibales in this project, the performance of each team member, the performance of each team and the decision of which 5 players will be chosen by their team to fight for the match. The first and the third random variables are independent of each other, while the second one is depended on the first one.

**1. The performance of each team member:**</br>

This random variable is represented by X_player and subjects to normal distribution.</br>
   
X_player = w1 * feature1 + w2 * feature2 +..., where features are rewards and punishments(such as shot and foul) of a player in a match, and w are the weights to measure how much contribution a feature makes to a player's comprehensive performance, and we have sum(w) = 1.</br>

In general, it is reasonable to assume that a player's performance subjects to normal distribution because in a certain period his performance is shifting around his average level and rarely has extreme situations. Hence, we will collect historical data of recent matches to calculate the expection and variance of X_player and get his performance distribution.

**2. The performance of each team:**</br>

This random varibale is represented by X_team = X_player1 + X_player2 + X_player3 + X_player4 + X_player5. (A team will choose 5 players to play the match). X_team subjects to normal distribution.</br>
  
Assume that each player's performance is independent of each other, since X_player subjects to normal distribution, then the mapping value, namely X_team, of their linear function also subjects to normal distribution. In simulation, the team who has a higher X_team will win the match.

**3. Decision of which 5 players will be chosen to fight for their team:**</br>

This random variable is represented by D_team and subjects to general discrete distribution.</br>
   
Ususally, the longer a player plays in recent matches, the higher probabilty that he will be chosen to fight for the upcoming match. Therefore, for each player in each team, we will calculate his probability of being chosen, and then for each team, we will randomly select 5 among 15 players according to the probabilities we get above and assign the decision to D_team

## Hypothesis or hypotheses before running the simulation:
1. Players are independent of each other.</br>
2. The performance of player is subject to normal distribution. (The reason has been stated above)</br>
3. The longer a player plays in recent matches, the higher the probability that he will be chose to play the upcoming match.</br>
4. After the simulation, the team who has highest probability of winning, namely who wins the most times in the simulation, will be the predicted winner.</br>

## Analytical Summary of your findings: (e.g. Did you adjust the scenario based on previous simulation outcomes?  What are the management decisions one could make from your simulation's output, etc.)
We ran the simulation for 15000 times and get the result as follows:</br>
Team name: Pelicans: 35.32 percent winning probability.</br>
Team name: Jazz: 22.02 percent winning probability.</br>
Team name: Thunders: 8.35 percent winning probability.</br>
Team name: Rockets: 8.35 percent winning probability.</br>
Team name: Trailblazers: 7.97 percent winning probability.</br>
Team name: Timberwolves: 7.84 percent winning probability.</br>
Team name: Warriors: 5.91 percent winning probability.</br>
Team name: Spurs: 4.25 percent winning probability.</br>

Now the match has came to the second round, and there are four teams in the second round:</br>
Jazz Pelicans Warriors Rockets </br>
Three of these four teams are in our predicted top 4 teams.


## Instructions on how to use the program:
Download all the files (including data files) and run the main_mc.py file

## All Sources Used:
Using Monte Carlo Modeling for Betting. https://www.pinnacle.com/en/betting-articles/Betting-Strategy/monte-carlo-betting-model/LC6JUNGE2GFTDWVP

