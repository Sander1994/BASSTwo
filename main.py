from Modell import *


def main():
    # model_learn(net_arch=[512, 512, 512, 512], total_timesteps=1000000)
    model_predict(n_envs=2, n_steps=22222)


if __name__ == "__main__":
    main()
