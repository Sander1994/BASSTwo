from Modell import *


def main():
    model_learn(net_arch=[2048, 2048, 2048, 2048, 2048], total_timesteps=1000000, ent_coef=0.3)
    model_predict(n_envs=1, render=True, n_steps=1000)


if __name__ == "__main__":
    main()
