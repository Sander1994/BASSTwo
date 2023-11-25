from Modell import *


def main():
    #model_learn(net_arch=[1024, 1024, 1024, 1024, 1024], total_timesteps=1110000,
    #            ent_coef=0.1, gamma=1, model_name="maskableppo_ganzschoenclever_193avg_v3")
    model_predict(n_envs=1, render=True, n_steps=40000, model_name="maskableppo_ganzschoenclever")


if __name__ == "__main__":
    main()
