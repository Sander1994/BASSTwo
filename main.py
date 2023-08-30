from Modell import *


def main():
    model_learn(net_arch=[512, 512, 512, 512], total_timesteps=500000)
    model_predict(n_envs=2)


if __name__ == "__main__":
    main()
