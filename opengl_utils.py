""" 
Copyright (C) Cortic Technology Corp. - All Rights Reserved
Written by Michael Ng <michaelng@cortic.ca>, 2021
"""

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
import pywavefront
import numpy as np

def Axes():
    glLineWidth(2)
    glBegin(GL_LINES)
    glColor3fv((1, 0, 0))
    glVertex3fv((0,0,0))
    glVertex3fv((1,0,0))
    glColor3fv((0, 1, 0))
    glVertex3fv((0,0,0))
    glVertex3fv((0,1,0))
    glColor3fv((0, 0, 1))
    glVertex3fv((0,0,0))
    glVertex3fv((0,0,1))
    glEnd()
    glLineWidth(1)
    glColor3fv((1, 1, 1))
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    gluSphere(gluNewQuadric(),0.5,20,20)

def surfaceToTexture(pygame_surface, texID):
    rgb_surface = pygame.image.tostring( pygame_surface, 'RGB')
    glBindTexture(GL_TEXTURE_2D, texID)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    surface_rect = pygame_surface.get_rect()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, surface_rect.width, surface_rect.height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_surface)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

def draw_camera_frame(surface, texID):
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)
    surfaceToTexture(surface, texID)
    glBindTexture(GL_TEXTURE_2D, texID)
    glBegin(GL_QUADS)
    glColor3fv((1, 1, 1))
    glTexCoord2f(0, 0); glVertex2f(-1, 1)
    glTexCoord2f(0, 1); glVertex2f(-1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, 1)
    glEnd()

def draw_axes(info, rotation_x, rotation_y, rotation_z):
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_LINE_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (info.current_w / info.current_h), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glPushMatrix()
    glTranslatef(-6.3, -2.5, -10)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)
    glRotatef(rotation_z, 0, 0, 1)
    Axes()
    glPopMatrix()

def load_model(model_path):
    scene = pywavefront.Wavefront(model_path, collect_faces=True)
    scene_box = (scene.vertices[0], scene.vertices[0])
    for vertex in scene.vertices:
        min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
        max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
        scene_box = (min_v, max_v)

    scene_size     = [scene_box[1][i]-scene_box[0][i] for i in range(3)]
    max_scene_size = max(scene_size)
    scaled_size    = 5
    scene_scale    = [scaled_size/max_scene_size for i in range(3)]
    scene_trans    = [-(scene_box[1][i]+scene_box[0][i])/2 for i in range(3)]
    
    return scene, scene_scale, scene_trans


def create_display_list(model_path):
    index = glGenLists(1)
    scene, scene_scale, scene_trans = load_model(model_path)
    glNewList(index, GL_COMPILE)
    glPushMatrix()
    glScalef(*scene_scale)
    glTranslatef(*scene_trans)

    for mesh in scene.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_i in face:
                normal = np.array(scene.parser.normals[vertex_i])
                glNormal3fv(normal)
                vertex = np.array(scene.vertices[vertex_i])
                glVertex3fv(vertex)
        glEnd()
    glPopMatrix()
    glEndList()
    index_2 = glGenLists(1)
    glNewList(index_2, GL_COMPILE)
    glPushMatrix()
    glScalef(*scene_scale)
    glTranslatef(*scene_trans)
    for mesh in scene.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            vertex_0 = np.array(scene.vertices[face[0]])
            vertex_1 = np.array(scene.vertices[face[1]])
            vertex_2 = np.array(scene.vertices[face[2]])
            vector_1 = vertex_1 - vertex_0
            vector_2 = vertex_2 - vertex_0
            normal = np.cross(vector_1, vector_2)
            normal = normal / np.linalg.norm(normal)
            for vertex_i in face:
                vertex = np.array(scene.vertices[vertex_i])
                glNormal3fv(normal)
                glVertex3fv(vertex)
        glEnd()
    glPopMatrix()
    glEndList()

    return index, index_2

def draw_model(info, 
               model_list, 
               rotation_x, 
               rotation_y, 
               rotation_z, 
               translation_x, 
               translation_y, 
               translation_z, 
               polygon_mode, 
               transparency_factor):

    if polygon_mode != "flat_wire":
        glPolygonMode(GL_FRONT_AND_BACK, polygon_mode)
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (info.current_w / info.current_h), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    if polygon_mode == GL_FILL or polygon_mode == "flat_wire":
        glEnable(GL_LIGHT1)
        glEnable(GL_LIGHT2)
    else:
        glDisable(GL_LIGHT1)
        glDisable(GL_LIGHT2)
    glEnable(GL_COLOR_MATERIAL)
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glPushMatrix()
    glTranslatef(0.0, 0.0, -15)
    glTranslatef(translation_x, translation_y, translation_z)
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)
    glRotatef(rotation_z, 0, 0, 1)
    glColor4fv((1, 1, 1, transparency_factor))
    if polygon_mode != "flat_wire":
        glCallList(model_list[0])
    else:
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(2.5, 2.5)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.0)
        glColor4f(0.0, 1.0, 0.0, transparency_factor)
        glCallList(model_list[1])
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING )
        glColor4f(1.0, 1.0, 1.0, transparency_factor)
        glCallList(model_list[1])
        glPopAttrib()
    glPopMatrix()
    glDisable(GL_LIGHT0)
    glDisable(GL_LIGHTING)
    glDisable(GL_COLOR_MATERIAL)