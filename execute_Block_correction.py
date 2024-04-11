from __future__ import annotations
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from ultralytics import YOLO
from typing import Iterable
import cv2
import pymssql
from datetime import datetime
import os
import numpy as np
import pandas as pd
from PIL import Image
import io
import hashlib

from theme_dropdown import create_theme_dropdown


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, *primary_50 10px, "
            #                      "*primary_50 20px)",
            body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, "
                                 "*primary_200 10px, *primary_200 20px)",
            body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, "
                                      "*primary_900 10px, *primary_900 20px)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )


# examples = [["dataset/valid/images/2_valid.jpg", "Image1"],
#             ["dataset/valid/images/3_valid.jpg", "Image2"],
#             ["dataset/valid/images/5_valid.jpg", "Image3"],
#             ["dataset/valid/images/8_valid.jpg", "Image4"]]


def authenticate(username, password):
    connection = pymssql.connect(
        host="",
        user="",
        password="",
        database=""
    )
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM detect_pipe_user WHERE username=%s", (username,))
    user_data = cursor.fetchone()

    connection.commit()
    connection.close()

    # check user exist
    if user_data:
        PK_ID, stored_username, stored_password, full_name, job_title = user_data  # table format
        # 將使用者輸入的密碼進行SHA2加密，並與資料庫儲存已加密過的密碼做判斷
        encrypted_input_password = encrypt_password(password)
        if username == stored_username and encrypted_input_password == stored_password:
            return True
    return False


def encrypt_password(password):
    # 對輸入的密碼進行 SHA2 加密
    encrypted_password = hashlib.sha256(password.encode()).hexdigest()
    return encrypted_password


# insert data to MS SQL Server
def insert_data_to_sql(image, result, total, label_txt, username):
    # BLOB format

    connection = pymssql.connect(
        host="",
        user="",
        password="",
        database=""
    )
    cursor = connection.cursor()

    table_name = 'detect_data'
    # if not cursor.tables(table=table_name).fetchone():
    #     create_table_query = f'''
    #         CREATE TABLE {table_name} (
    #             id INT PRIMARY KEY IDENTITY(1,1),
    #             datetime DATETIME,
    #             image_BLOB IMAGE,
    #             detect_image_BLOB IMAGE,
    #             username NVARCHAR(255),
    #             total INT
    #         )
    #     '''
    #     cursor.execute(create_table_query)

    # 插入数据
    insert_query = f"INSERT INTO {table_name} (datetime, image_BLOB, detect_image_BLOB, label, username, total) VALUES (%s, %s, %s, %s, %s, %s)"
    cursor.execute(insert_query, (datetime.now(), image, result, label_txt, username, total))

    connection.commit()
    connection.close()


def detect_objects_on_image(image_path, username: gr.Request):
    image = cv2.imread(image_path)
    with open(image_path, 'rb') as File:
        BinaryData = File.read()
    model = YOLO("runs/detect/train10/weights/best.pt")
    results = model.predict(image_path, save_txt=True, device='0', conf=0.51, project='detect_label', exist_ok=True)
    '''
    reference: https://docs.ultralytics.com/usage/cfg/#predict
    
    Key	                Value	                Description
    source	            'ultralytics/assets'	source directory for images or videos
    conf	            0.25	                object confidence threshold for detection
    iou	                0.7	                    intersection over union (IoU) threshold for NMS
    half	            False	                use half precision (FP16)
    device	            None	                device to run on, i.e. cuda device=0/1/2/3 or device=cpu
    show	            False	                show results if possible
    save	            False	                save images with results
    save_txt	        False	                save results as .txt file
    save_conf	        False	                save results with confidence scores
    save_crop	        False	                save cropped images with results
    show_labels	        True	                show object labels in plots
    show_conf	        True	                show object confidence scores in plots
    max_det	            300	                    maximum number of detections per image
    vid_stride	        False	                video frame-rate stride
    stream_buffer	    bool	                buffer all streaming frames (True) or return the most recent frame (False)
    line_width	        None	                The line width of the bounding boxes. If None, it is scaled to the image size.
    visualize	        False	                visualize model features
    augment	            False	                apply image augmentation to prediction sources
    agnostic_nms	    False	                class-agnostic NMS
    retina_masks	    False	                use high-resolution segmentation masks
    classes	            None	                filter results by class, i.e. classes=0, or classes=[0,2,3]
    boxes	            True	                Show boxes in segmentation predictions
    '''
    result = results[0]
    output = []
    temp_total = 1000
    total = 0
    aligns = image.shape
    align_bottom = aligns[0]
    align_right = (aligns[1] / 1.7)
    font_scale = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    a_temp = f"pipe = {temp_total}"
    text_size, _ = cv2.getTextSize(a_temp, font, font_scale, font_thickness)
    text_width, text_height = text_size
    background_rect_x = int(align_right)
    background_rect_y = int(align_bottom) - text_height - 5  # vertical position
    background_rect_width = text_width
    background_rect_height = text_height
    background_color = (0, 255, 255)
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()

        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color=(36, 255, 12),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        total += 1  # calculate the total detection

        # cv2.putText(image, result.names[class_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)   # show label class name on top of rectangle

    # paint total layer
    cv2.rectangle(image,
                  (background_rect_x, background_rect_y),
                  (background_rect_x + background_rect_width, background_rect_y + background_rect_height),
                  background_color, -1)

    a = f"PIPE = {total}"

    cv2.putText(image, a, (int(align_right), int(align_bottom - 5)), font, font_scale, (255, 0, 0), font_thickness,
                cv2.LINE_AA)  # justify background layer

    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_result_encode = cv2.imencode('.jpg', image)[1]
    data_result_encode = np.array(img_result_encode)
    byte_result_encode = data_result_encode.tobytes()

    # txt save binary success, checked decode success
    file_path = 'detect_label/predict/labels/' + str(global_filename) + '.txt'

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as file:
            label_txt = file.read()
        os.remove(file_path)
    else:
        label_txt = b'None'

    user = username.username

    # insert data to MS SQL Server
    insert_data_to_sql((BinaryData, ), byte_result_encode, total, label_txt, user)

    global global_total
    global_total = total

    return image_RGB


def change_flagbox(sub):
    return gr.Button(interactive=True), gr.Number(visible=False), gr.Button(visible=False)


def change_backflagbox(sub):
    return gr.Button(interactive=False), gr.Number(visible=False), gr.Button(visible=False), gr.Button(interactive=False)


def change_out_backflagbox(sub):
    return gr.Button(interactive=False), gr.Number(visible=False), gr.Button(visible=False), gr.Button(interactive=False)


def change_txtbox(sub):
    return gr.Number(visible=True, value=global_total), gr.Button(visible=True)


def change_confirmbox(sub):
    return gr.Dropdown(value=None), gr.Image(value=None)


def change_dropdownbox(sub):
    return gr.Dropdown(interactive=True)


def change_subbtn(files):
    file_name_with_extension = os.path.basename(files)
    file_name, extension = os.path.splitext(file_name_with_extension)
    global global_filename
    global_filename = file_name

    input_image = cv2.imread(files)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    return gr.Button(interactive=True), input_image


def return_value(x, username: gr.Request):
    connection = pymssql.connect(
        host="",
        user="",
        password="",
        database=""
    )
    cursor = connection.cursor()

    table_name = 'detect_data'
    update_query = f'''
        UPDATE {table_name}
        SET total = %s,
            flag = 1
        WHERE id = (
            SELECT TOP 1 id
            FROM {table_name}
            WHERE username = %s
            ORDER BY id DESC
        )
    '''
    user = username.username
    cursor.execute(update_query, (x, user))
    connection.commit()
    connection.close()

    gr.Info("數據已更新完成，請無視!")


def highlight_cols(dataframe):
    df = dataframe.copy()
    df.loc[:, :] = 'color: green'
    # df[['時間', '總數', '異常狀態']] = 'color: purple'
    return df


def search_value(sub, usernmame: gr.Request):
    connection = pymssql.connect(
        host="10.3.96.168",
        user="N000184123",
        password="N000184123@npc",
        database="3033"
    )
    cursor = connection.cursor()
    table_name = 'detect_data'
    # table_name_2 = 'detect_pipe_user'
    user = usernmame.username
    if user != 'admin':
        search_query = f'''
            SELECT id, CONVERT(VARCHAR(8), datetime, 8) AS time_only, total, flag, username
            FROM {table_name}
            WHERE CONVERT(DATE, datetime) = '{sub}'
            AND username = '{user}'
        '''
        # search_query = f'''
        #     SELECT t1.id, CONVERT(VARCHAR(8), t1.datetime, 8) AS time_only, t1.total, t1.flag, t2.full_name
        #     FROM {table_name} t1
        #     JOIN {table_name_2} t2 ON t1.username = t2.username
        #     WHERE CONVERT(DATE, t1.datetime) = '{sub}'
        #     AND t1.username = '{user}';
        # '''
    else:
        search_query = f'''
            SELECT id, CONVERT(VARCHAR(8), datetime, 8) AS time_only, total, flag, username
            FROM {table_name}
            WHERE CONVERT(DATE, datetime) = '{sub}'
        '''
        # search_query = f'''
        #     SELECT t1.id, CONVERT(VARCHAR(8), t1.datetime, 8) AS time_only, t1.total, t1.flag, t2.full_name
        #     FROM {table_name} t1
        #     JOIN {table_name_2} t2 ON t1.username = t2.username
        #     WHERE CONVERT(DATE, datetime) = '{sub}'
        # '''
    cursor.execute(search_query)
    results = cursor.fetchall()
    connection.close()

    # connection = pymssql.connect(
    #     host="",
    #     user="",
    #     password="",
    #     database=""
    # )
    # cursor = connection.cursor()
    # table_name_2 = 'detect_pipe_user'
    #
    # search_same_name = f'''
    # SELECT t1.id, t1.時間, t1.總數, t1.異常狀態, t2.full_name
    # FROM {table_name_2} t1
    # JOIN {table_name} t2 ON t1.使用者 = t2.user_id;'''

    columns = ['id', '時間', '總數', '異常狀態', '使用者']

    pd_result = pd.DataFrame(results, columns=columns)

    search_result_df = pd_result.style.apply(highlight_cols, axis=None)

    return search_result_df


def search_id_list(sub, user):
    connection = pymssql.connect(
        host="",
        user="",
        password="",
        database=""
    )
    cursor = connection.cursor()
    table_name = 'detect_data'
    # user = username.username
    if user != 'admin':
        search_query = f'''
            SELECT id, CONVERT(VARCHAR(8), datetime, 8) AS time_only, total, flag, username
            FROM {table_name}
            WHERE CONVERT(DATE, datetime) = '{sub}'
            AND username = '{user}'
        '''
    else:
        search_query = f'''
            SELECT id, CONVERT(VARCHAR(8), datetime, 8) AS time_only, total, flag, username
            FROM {table_name}
            WHERE CONVERT(DATE, datetime) = '{sub}'
        '''
    cursor.execute(search_query)
    results = cursor.fetchall()
    connection.close()

    columns = ['id', '時間', '總數', '異常狀態', '使用者']

    pd_result = pd.DataFrame(results, columns=columns)

    id_list = pd_result['id'].to_list()

    return id_list


def Dropdown_list(x, username: gr.Request):
    user = username.username
    dd = gr.Dropdown(search_id_list(x, user))
    return dd

def update_image_dropdown(x):
    if not x:  # 檢查 image_id 是否為空
        return None  # 如果是空的話直接返回 None

    image_id = x

    connection = pymssql.connect(
        host="",
        user="",
        password="",
        database=""
    )

    cursor = connection.cursor()

    cursor.execute(f"select detect_image_BLOB from detect_data WHERE id = '{image_id}'")
    image_blob = cursor.fetchone()[0]

    image = Image.open(io.BytesIO(image_blob))

    cursor.close()
    connection.close()

    return image


def create_greeting(username: gr.Request):
    return gr.Markdown(value=f"## 目前使用者為: {username.username}")


def create_greeting_1(username: gr.Request):
    return gr.Markdown(value=f"## 目前使用者為: {username.username}")


def get_current_date():
    return datetime.now().date()


# def logout(username: gr.Request):
#     del username.username
#     print("logged out")
#     return


seafoam = Seafoam()

dropdown, js = create_theme_dropdown()

with gr.Blocks(theme=seafoam) as detect_demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("")
            gr.Markdown('')
        with gr.Column():
            user_show = gr.Markdown(value="No User")
            toggle_dark_p1 = gr.Button(value="淺色/深色模式")
    detect_demo.load(create_greeting, inputs=None, outputs=user_show)
        #     logout_button = gr.LogoutButton(value="Logout", icon=None)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type='numpy', label='上傳圖片')
            with gr.Row():
                clear_btn = gr.ClearButton(inp, value='清空')
                upload_button = gr.UploadButton("點選圖片", file_types=["image"])
                submit_btn = gr.Button(value='硬管數量辨識', interactive=False)
        with gr.Column():
            out = gr.Image(type='numpy', label='偵測結果')
            submit_btn.click(detect_objects_on_image, inputs=upload_button, outputs=out)
            with gr.Row():
                clear_out_btn = gr.ClearButton([inp, out], value='清空所有')
                flag = gr.Button(value='數量錯誤請點選', interactive=False)
                count_total = gr.Number(visible=False, label='實際總數:')
                submit_total = gr.Button(value='上傳更新數量', visible=False)

    # logout_button.click(logout, inputs=logout_button, outputs=None)

    upload_button.upload(change_subbtn, inputs=upload_button, outputs=[submit_btn, inp])  # 需修改

    submit_btn.click(change_flagbox, inputs=submit_btn, outputs=[flag, count_total, submit_total])

    flag.click(change_txtbox, inputs=flag, outputs=[count_total, submit_total])

    clear_btn.click(change_backflagbox, inputs=clear_btn, outputs=[flag, count_total, submit_total, submit_btn])

    clear_out_btn.click(change_out_backflagbox, inputs=clear_out_btn,
                        outputs=[flag, count_total, submit_total, submit_btn])

    submit_total.click(return_value, inputs=count_total)

    toggle_dark_p1.click(
        None,
        js="""
            () => {
                document.body.classList.toggle('dark');
                document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)'
            }
            """,
    )


with gr.Blocks(theme=seafoam) as demonstrate_demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("")
            gr.Markdown('')
        with gr.Column():
            user_show_second = gr.Markdown(value="No User")
            toggle_dark_p2 = gr.Button(value="淺色/深色模式")
    demonstrate_demo.load(create_greeting_1, inputs=None, outputs=user_show_second)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                # current_date = datetime.now().date()
                # dt = gr.HTML(f"""<input type="date" id="date" name="date" value="{current_date}"
                # max="{current_date}">""")
                dt = gr.HTML(f"""<input type="date" id="date" name="date" value="{get_current_date()}"
                                                                            max="{get_current_date()}">""")
                search_btn = gr.Button(value='確認')
            # with gr.Row():
                x = gr.Textbox(label='搜尋日期為:', value='YYYY-MM-DD', interactive=False, visible=False)
                # confirm_btn = gr.Button(value='確認搜尋', interactive=False)
            show_result = gr.DataFrame()
        with gr.Column():
            dropdown = gr.Dropdown(interactive=False, label='選擇檢視圖片id')
            out_result = gr.Image(interactive=False)

    # search_btn.click(None, None, x, js='(x) => {return (document.getElementById("date")).value;}')

    search_btn.click(change_confirmbox, inputs=search_btn, outputs=[dropdown, out_result])

    search_btn.click(search_value, inputs=x, outputs=show_result,
                      js='(x) => {return (document.getElementById("date")).value;}')

    search_btn.click(change_dropdownbox, inputs=search_btn, outputs=dropdown)

    search_btn.click(Dropdown_list, inputs=x, outputs=dropdown,
                      js='(x) => {return (document.getElementById("date")).value;}')

    dropdown.input(update_image_dropdown, inputs=dropdown, outputs=out_result)

    toggle_dark_p2.click(
        None,
        js="""
                () => {
                    document.body.classList.toggle('dark');
                    document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)'
                }
                """,
    )


app = gr.TabbedInterface([detect_demo, demonstrate_demo], ["偵測圖片", "檢測圖片"], theme=seafoam)

if __name__ == "__main__":
    app.launch(server_name='0.0.0.0',
               server_port=80,
               auth=authenticate,
               auth_message='PleaseEnter USERNAME & PASSWORD',)
               # inbrowser=True)

