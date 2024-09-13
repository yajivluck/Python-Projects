import cv2
import argparse
from Optimizer import Optimizer
from Player import MsPacManGame


def get_args():
    """Gets parsed command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="plays Ms. Pac-Man")
    parser.add_argument("--no-learn", default=False, action="store_true",
                        help="play without training")
    parser.add_argument("--episodes", default=1, type=int,
                        help="number of episodes to run")
    parser.add_argument("--learning-rate", default=0.01, type=float,
                        help="learning rate")
    parser.add_argument("--no-display", action="store_false", default=True,
                        help="do not display the game on the screen (faster)",
                        dest="display")
    parser.add_argument("--map-display", action="store_true", default=False,
                        help="whether to display the map on screen or not")
    parser.add_argument("--seed", default=None, type=int,
                        help="seed for random number generator to use")

    return parser.parse_args()


if __name__ == "__main__":
    
    
    average_reward_per_episode = []
    

    args = get_args()
    game = MsPacManGame(args.seed, args.display)
    #game = MsPacManGame(args.seed, False)

    
    agent = Optimizer(args.learning_rate)
    
    total_rewards = 0
    min_rewards = float("inf")
    max_rewards = 0
    #for episode in range(args.episodes):
    for episode in range(50):
        while not game.game_over():
            #print("Episode {}: {}".format(episode + 1, game.reward))
            # if episode:
            #     print("Average: {}".format(total_rewards / episode))
                
            max_rewards = max(max_rewards, game.reward)
            #print("Max: {}".format(max_rewards))
            #print("Min: {}".format(min_rewards))

            prev_state = game.sliced_map.map
            optimal_a, expected_utility = agent.get_optimal_action(game)
            reward = game.act(optimal_a)

    
            #TODO for no learning
            if not args.no_learn:
                #print(agent.formatted_weights())
                agent.update_weights(prev_state, optimal_a, game,
                                     expected_utility, reward)
                
            #TODO MODIFY THIS MANUALLY (REMOVE OR TRUE)
            #if args.map_display or True:
            if args.map_display:

                game_map = game.map
                sliced_game_map = game.sliced_map

                cv2.imshow("map", game_map.to_image())
                cv2.imshow("sliced map", sliced_game_map.to_image())
                cv2.waitKey(1)

        print("Episode Complete {}: {}".format(episode + 1, game.reward))
        print("GLIE: {}".format(agent.glie))
        min_rewards = min(min_rewards, game.reward)
        total_rewards += game.reward
        
        
        print('average:', total_rewards/ (episode + 1 ))
        average_reward_per_episode.append(total_rewards/ (episode + 1))

        #if not args.no_learn:
            #agent.save()

        game.reset_game()
