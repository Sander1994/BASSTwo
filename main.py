from stable_baselines3 import PPO
from GanzSchoenCleverEnv import GanzSchonCleverEnv


def train_and_test_model():
    env = GanzSchonCleverEnv()  # Here we directly create one instance of the environment

    model = PPO("MlpPolicy", env, verbose=1)  # We pass the single environment instance to the model
    model.learn(total_timesteps=250)
    model.save("ppo_ganzschoenclever")

    del model

    model = PPO.load("ppo_ganzschoenclever")

    obs = env.reset()  # Use the single environment instance to reset and get the initial observation
    while True:
        try:
            action, _states = model.predict(obs)
        except ValueError as e:
            print("Observation that caused error:", obs)
            raise e
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render("human")  # Use the single environment instance to render


def main():
    # Other parts of your code

    # At some point you decide to start the training and testing of the model
    train_and_test_model()

    # More of your code


if __name__ == "__main__":
    main()
