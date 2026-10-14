#程序入口
from agent.core import Agent


def main():
    agent = Agent()

    print("Agent 已启动，输入 exit 退出。")

    while True:
        user_input = input("\n你：").strip()

        if user_input.lower() == "exit":
            print("程序已退出。")
            break

        if not user_input:
            continue

        try:
            answer = agent.run(user_input)
            print(f"\nAgent：{answer}")
        except Exception as exc:
            print(f"\n程序运行出错：{exc}")


if __name__ == "__main__":
    main()
