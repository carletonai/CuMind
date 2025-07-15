from cumind.utils.config import config


def main() -> None:
    config.to_json("configuration.json")


if __name__ == "__main__":
    main()
