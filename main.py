""" 
Copyright (C) Cortic Technology Corp. - All Rights Reserved
Written by Michael Ng <michaelng@cortic.ca>, 2021
"""

import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
import pygame_gui
from ctypes import *
import math
from pygame.locals import *
from HandTracker import HandTracker
from utils import circularlist, draw_object_imgs, draw_hand_landmarks, draw_zoom_scale
from opengl_utils import draw_axes, draw_camera_frame, create_display_list, draw_model

SCREEN_WIDTH = 1152
SCREEN_HEIGHT = 648

# These are values I measured for my hand and camera setup, you can change them to your own settings
max_finger_distance = 225
min_finger_distance = 25
middle_finger_distance = (max_finger_distance - min_finger_distance) / 2
max_scale = 11

max_x_plane_angle = 60
max_y_plane_angle = 30
middle_y_plane_angle = 90
middle_x_plane_angle = 70

current_finger_distance = circularlist(size=7)

single_handed = False
zoom_mode = False
zoom_mode_count = 0
rotate_mode = False
rotate_mode_count = 0
cursor_mode = False
cursor_mode_count = 0

plane_x_angle = 0
previous_plane_x_angle = 0
plane_z_angle = 0
previous_plane_z_angle = 0
plane_y_angle = 0
previous_plane_y_angle = 0

tracker = HandTracker(
        input_src=None, 
        use_lm=True, 
        stats=True,
        trace=False
        )


# You can load in your own 3d models in obj format
model_name_list = {
                   "Bunny": "./3d_models/bunny.obj"
                #    "Elephant": "./3d_models/elephant.obj",
                #   "Dragon": "./3d_models/dragon.obj"
                }

current_model = "Bunny"

current_selected_button = ""

cursor_img = cv2.imread("./resources/cursor.png", cv2.IMREAD_UNCHANGED)
cursor_img = cv2.resize(cursor_img, (30, 30))

transparency_factor = 0.6

def main():
    global model_name_list
    global current_model
    global current_selected_button
    global plane_x_angle
    global previous_plane_x_angle
    global plane_z_angle
    global previous_plane_z_angle
    global plane_y_angle
    global previous_plane_y_angle
    global current_finger_distance
    global single_handed
    global zoom_mode
    global zoom_mode_count
    global rotate_mode
    global rotate_mode_count
    global cursor_mode
    global cursor_mode_count


    # Initial values of model rotation, translation, and polygon.
    rotation_x = 0
    rotation_y = 0
    rotation_z = 0
    translation_x = 0
    translation_y = 0
    translation_z = 0
    polygon_mode = GL_LINE
    
    # Use PyGame to create an OpenGL drawing surface
    pygame.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), OPENGL | DOUBLEBUF)
    pygame.display.init()
    info = pygame.display.Info()

    texID = glGenTextures(1)
    
    # Create a surface to draw the camera frame
    offscreen_surface = pygame.Surface((info.current_w, info.current_h))
    
    # Use PyGame GUI to draw UI elements
    manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT), 'theme.json')
    
    wire_frame_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((226, 50), (200, 60)),
                                            text='Wireframe',
                                            manager=manager)
    flat_shading_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((476, 50), (200, 60)),
                                            text='Flat Shading',
                                            manager=manager)
    smooth_shading_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((726, 50), (200, 60)),
                                            text='Smooth Shading',
                                            manager=manager)
    
    # Load the 3d models into memory for future drawing
    for model in model_name_list:
        model_path = model_name_list[model]
        index, index_2 = create_display_list(model_path)
        model_name_list[model] = [index, index_2]

    # Setup OpenGL scene and lightings
    glViewport(0, 0, info.current_w, info.current_h)
    
    glLight(GL_LIGHT0, GL_POSITION,  (0, 0, 1, 0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))

    glLight(GL_LIGHT1, GL_POSITION,  (1, 1, 0, 0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (0, 0, 1, 1.0))

    glLight(GL_LIGHT2, GL_POSITION,  (-1, 1, 0, 0))
    glLightfv(GL_LIGHT2, GL_DIFFUSE, (1, 0, 0, 1.0))
    
    glEnable(GL_NORMALIZE)
    glEnable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)
    
    clock = pygame.time.Clock()
    
    done = False
    
    while not done:
        if cursor_mode:
            wire_frame_button.show()
            flat_shading_button.show()
            smooth_shading_button.show()
            if current_selected_button == "Wire":
                wire_frame_button.select()
            if current_selected_button == "Smooth":
                smooth_shading_button.select()
            if current_selected_button == "Flat":
                flat_shading_button.select()

        else:
            wire_frame_button.hide()
            flat_shading_button.hide()
            smooth_shading_button.hide()

        time_delta = clock.tick(60)/1000.0

        # Handle key press events
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            rotation_y += 2
        if keys[pygame.K_RIGHT]:
            rotation_y -= 2
        if keys[pygame.K_UP]:
            rotation_x += 2
        if keys[pygame.K_DOWN]:
            rotation_x -= 2
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
            if (event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED):
                if event.ui_element == wire_frame_button:
                    polygon_mode = GL_LINE
                if event.ui_element == smooth_shading_button:
                    polygon_mode = GL_FILL
                if event.ui_element == flat_shading_button:
                    polygon_mode = "flat_wire"
            # Allowing mouse scroll wheel to perform zoom in and zoom out
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    translation_z += 1
                if event.button == 5:
                    translation_z -= 1
            manager.process_events(event)
        manager.update(time_delta)

        # Clear the OpenGL buffers everytime before drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Obtain camera frame and hand landmarks
        frame, hands, _ = tracker.next_frame()

        plane_y_angle = middle_y_plane_angle
        plane_x_angle = middle_x_plane_angle

        # Only enable 3d rotation manipulation when both hands are present
        if len(hands) == 2:
            rotate_mode = True
            zoom_mode_count = 0
            cursor_mode_count = 0
            zoom_mode = False
            cursor_mode = False
            single_handed = False
            if hands[0].handedness > 0.5:
                thumb1 = np.array(hands[0].landmarks[4])
                index1 = np.array(hands[0].landmarks[8])
                index2 = np.array(hands[1].landmarks[8])
            else:
                thumb1 = np.array(hands[1].landmarks[4])
                index1 = np.array(hands[1].landmarks[8])
                index2 = np.array(hands[0].landmarks[8])

            # The 3 rotational angles are calculated by measuring the orientation of the plane (or line)
            # connecting the 2 hands, using the index fingers in this program
            z_radians = math.atan2(index2[1] - index1[1], index2[0]-index1[0])
            y_radians = math.atan2(index2[0] - index1[0], index2[2]-index1[2])
            x_radians = math.atan2(thumb1[1] - index1[1], thumb1[2]-index1[2])
            plane_z_angle = math.degrees(z_radians)
            # Check to make sure the direction of angle is consistent across all frames
            if abs(plane_z_angle - previous_plane_z_angle) > 100:
                z_radians = math.atan2(index1[1] - index2[1], index1[0]-index2[0])
                plane_z_angle = math.degrees(z_radians)
            previous_plane_z_angle = plane_z_angle
            plane_y_angle = math.degrees(y_radians)
            # Check to make sure the direction of angle is consistent across all frames
            if abs(plane_y_angle - previous_plane_y_angle) > 100:
                y_radians = math.atan2(index1[0] - index2[0], index1[2]-index2[2])
                plane_y_angle = math.degrees(y_radians)
            previous_plane_y_angle = plane_y_angle
            plane_x_angle = math.degrees(x_radians)
            # Check to make sure the direction of angle is consistent across all frames
            if abs(plane_x_angle - previous_plane_x_angle) > 100:
                x_radians = math.atan2(index1[1] - index2[1], index1[2]-index2[2])
                plane_x_angle = math.degrees(x_radians)
            previous_plane_x_angle = plane_x_angle
        elif len(hands) == 1:
            rotate_mode = False
            rotate_mode_count = 0
            single_handed = True
            thumb = np.array(hands[0].landmarks[4])
            index = np.array(hands[0].landmarks[8])
            # In one hand mode, use gesture recognition to decide whether user wants
            # to zoom in/out of a model, or user wants to move the cursor to select 
            # a different rednering mode
            if hands[0].gesture == "ZOOM":
                # Use a counter to filter out falsely recognized gesture
                zoom_mode_count += 1
                if zoom_mode_count > 4:
                    zoom_mode = True
                    cursor_mode_count = 0
                    cursor_mode = False
                    finger_distance = np.linalg.norm(thumb - index)
                    current_finger_distance.append(finger_distance)
            elif hands[0].gesture == "ONE":
                # Use a counter to filter out falsely recognized gesture
                cursor_mode_count += 1
                if cursor_mode_count > 4:
                    cursor_mode = True
                    zoom_mode_count = 0
                    zoom_mode = False
            cursor_x = index[0]
            cursor_y = index[1]
        else:
            # If no hand is in the scene, reset all mode
            rotate_mode = False
            zoom_mode = False
            cursor_mode = False
            rotate_mode_count = 0
            zoom_mode_count = 0
            cursor_mode_count = 0
            
        if single_handed and cursor_mode:
                draw_object_imgs(frame, cursor_img, int(cursor_x - cursor_img.shape[1] // 2), int(cursor_y - cursor_img.shape[0] // 2), int(cursor_x + cursor_img.shape[1] // 2), int(cursor_y + cursor_img.shape[0] // 2), 1)
                selected_button = None
                if wire_frame_button.hover_point((SCREEN_WIDTH-cursor_x), cursor_y):
                    wire_frame_button.select()
                    selected_button = "Wire"
                    polygon_mode = GL_LINE
                if smooth_shading_button.hover_point((SCREEN_WIDTH-cursor_x), cursor_y):
                    smooth_shading_button.select()
                    selected_button = "Smooth"
                    polygon_mode = GL_FILL
                if flat_shading_button.hover_point((SCREEN_WIDTH-cursor_x), cursor_y):
                    flat_shading_button.select()
                    selected_button = "Flat"
                    polygon_mode = "flat_wire"

                if selected_button != None and current_selected_button != "":
                    if selected_button != current_selected_button:
                        #print(selected_button, current_selected_button)
                        if current_selected_button == "Wire":
                            wire_frame_button.unselect()
                        if current_selected_button == "Smooth":
                            smooth_shading_button.unselect()
                        if current_selected_button == "Flat":
                            flat_shading_button.unselect()

                if selected_button is not None:
                    current_selected_button = selected_button

        # Calcuate the manipulation values for the model
        scale = 0
        relative_finger_distance = current_finger_distance.calc_average() - middle_finger_distance
        if relative_finger_distance >= 0:
            scale = max_scale * (relative_finger_distance / (max_finger_distance - middle_finger_distance))
        else:
            scale = -max_scale * (relative_finger_distance / (min_finger_distance - middle_finger_distance))

        if scale > max_scale:
            scale = max_scale
        if scale < -max_scale:
            scale = -max_scale

        translation_z = int(scale)

        if frame is not None:
            frame = draw_hand_landmarks(frame, hands, zoom_mode, single_handed)
            frame = cv2.flip(frame, 1)
            if zoom_mode:
                draw_zoom_scale(frame, translation_z, max_scale, SCREEN_HEIGHT)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.swapaxes(0, 1)
            frame = pygame.surfarray.make_surface(frame)
            offscreen_surface.blit(frame,(0,0))
            manager.draw_ui(offscreen_surface)
            
            draw_camera_frame(offscreen_surface, texID)

        rotation_z = int(plane_z_angle)
        rotation_y = int(90 * ((plane_y_angle - middle_y_plane_angle) / max_y_plane_angle))
        rotation_x = int(-90 * ((plane_x_angle - middle_x_plane_angle) / max_x_plane_angle))

        draw_model(info, 
                   model_name_list[current_model], 
                   rotation_x, 
                   rotation_y, 
                   rotation_z, 
                   translation_x, 
                   translation_y, 
                   translation_z, 
                   polygon_mode, 
                   transparency_factor)

        if rotate_mode:
            draw_axes(info, rotation_x, rotation_y, rotation_z)
            
        pygame.display.flip()
        clock.tick(60)

main()
pygame.quit()
