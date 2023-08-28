from Modell import *


def main():
    model_learn(net_arch=[512, 512, 512, 512], total_timesteps=100000)
    model_predict()


if __name__ == "__main__":
    main()
