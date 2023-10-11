from Modell import *


def main():
    model_learn(net_arch=[512, 512, 512, 512], total_timesteps=30000, ent_coef=0.1)
    model_predict(n_envs=1, render=True, n_steps=400)


if __name__ == "__main__":
    main()
