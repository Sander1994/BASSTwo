from Modell import *


def main():
    model_learn(net_arch=[1024, 1024, 1024, 1024, 1024], total_timesteps=1110000, ent_coef=0.1)
    model_predict(n_envs=1, render=True, n_steps=40000)


if __name__ == "__main__":
    main()
