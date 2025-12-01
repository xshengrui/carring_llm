import json
import os

def renumber_ids_and_combine_jsonl(folder_path, output_filename="renumbered_combined_data.jsonl"):
    """
    将指定文件夹下的所有JSON Lines文件的数据整合到一个新的JSON Lines文件中，
    并重新编号每个JSON对象的 'id' 字段，使其从1开始连贯计数。

    Args:
        folder_path (str): 包含JSON Lines文件的文件夹路径。
        output_filename (str): 输出的JSON Lines文件名，默认为 "renumbered_combined_data.jsonl"。
    """
    output_file_path = os.path.join(os.getcwd(), output_filename)
    global_index = 0
    total_processed = 0

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for filename in os.listdir(folder_path):
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(folder_path, filename)
                    print(f"正在处理文件: {filename}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            for line_num, line in enumerate(infile, 1):
                                try:
                                    data = json.loads(line.strip())
                                    # 重新设置 'id' 的值
                                    data['id'] = global_index
                                    json.dump(data, outfile, ensure_ascii=False)
                                    outfile.write('\n')
                                    global_index += 1
                                    total_processed += 1
                                except json.JSONDecodeError:
                                    print(f"警告: 文件 '{filename}' 第 {line_num} 行不是有效的JSON。已跳过此行。")
                                except KeyError:
                                    print(f"警告: 文件 '{filename}' 第 {line_num} 行的JSON不包含 'id' 字段。已跳过此行。")
                                except Exception as e:
                                    print(f"警告: 处理文件 '{filename}' 第 {line_num} 行时发生未知错误: {e}。已跳过此行。")
                    except FileNotFoundError:
                        print(f"错误: JSON Lines文件未找到: {filename}")
                    except Exception as e:
                        print(f"处理文件 {filename} 时发生未知错误: {e}")

        if total_processed > 0:
            print(f"\n已将所有JSON Lines文件的数据整合到: {output_file_path}")
            print(f"并重新编号了 {total_processed} 条数据的 'id'，起始ID为 0。")
        else:
            print("文件夹中没有找到任何JSON Lines文件。")

    except Exception as e:
        print(f"创建或写入输出文件 '{output_file_path}' 时发生错误: {e}")

if __name__ == "__main__":
    folder_path = "./data/pdf_train_data"
    output_filename = "./data/pdf_train_data/combined_data.jsonl"

    if not os.path.isdir(folder_path):
        print(f"错误: 提供的文件夹路径 '{folder_path}' 不存在或不是一个有效的文件夹。")
    else:
        renumber_ids_and_combine_jsonl(folder_path, output_filename)
