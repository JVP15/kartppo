from desmume.emulator import DeSmuME, SCREEN_PIXEL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_PIXEL_SIZE_BOTH, SCREEN_HEIGHT_BOTH
from desmume.controls import keymask, Keys
import cv2
import numpy as np


PLAYER_DATA_ADDR = 0x217ACF8
PLAYER_SPEED_OFFSET = 0x2A8

CHECKPOINT_DATA_ADDR = 0x021661B0
CHECKPOINT_OFFSET = 0xDAE

# look here to unlock stuff http://uk.codejunkies.com/search/codes/Mario-Kart-DS_Nintendo-DS_17825906-17___.aspx
# https://tasvideos.org/GameResources/DS/MarioKartDS

if __name__ == '__main__':
    emu = DeSmuME()
    mem = emu.memory

    emu.open('ROM/Mario Kart DS.nds')
    emu.volume_set(0)

    key_dict = {ord('j'): keymask(Keys.KEY_LEFT), ord('l'): keymask(Keys.KEY_RIGHT), ord('i'): keymask(Keys.KEY_UP),
                ord('k'): keymask(Keys.KEY_DOWN),
                ord('a'): keymask(Keys.KEY_A), ord('s'): keymask(Keys.KEY_B), ord('d'): keymask(Keys.KEY_X),
                ord('f'): keymask(Keys.KEY_Y)}

    i = 0

    while True:
        i += 1
        print(i)
        emu.cycle()
        emu.input.keypad_rm_key(Keys.NO_KEY_SET)

        if emu.memory.signed[0x223ce2e0] != 0x7f:
            emu.memory.write_byte(0x223ce2e0, 0x7f)
            print('unlocked all nitro courses')
        if emu.memory.signed[0x223ce2e1] != 0x7f:
            emu.memory.write_byte(0x223ce2e1, 0x7f)
            print('unlocked all retro courses')
        if emu.memory.signed[0x223ce2e2] != 0x7f:
            emu.memory.write_byte(0x223ce2e2, 0x7f)
            print('unlocked all characters')

        buff = emu.display_buffer_as_rgbx()

        gpu_framebuffer = np.frombuffer(buff, dtype=np.uint8)

        #upper_image = gpu_framebuffer[:SCREEN_PIXEL_SIZE*4]
        #lower_image = gpu_framebuffer[SCREEN_PIXEL_SIZE*4:]

        #upper_image = upper_image.reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 4))
        #lower_image = lower_image.reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 4))

        #cv2.imshow('frame', upper_image)
        #cv2.imshow('frame2', lower_image)

        whole_frame = gpu_framebuffer[:SCREEN_PIXEL_SIZE_BOTH * 4]
        whole_frame = whole_frame.reshape((SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4))
        # resize and make grayscale to 84x84 to match Atari
        #whole_frame = cv2.resize(whole_frame, (84, 84))
        #whole_frame = cv2.cvtColor(whole_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', whole_frame)

        pressed_key = cv2.waitKey(10) & 0xFF

        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('c'):
            track_name = input('enter track name:')
            if track_name:
                emu.savestate.save_file('ROM/linux_saves/' + track_name + '.dsv')
        elif pressed_key == ord('v'):
            track_name = input('enter track name:')
            if track_name:
                emu.savestate.load_file('ROM/linux_saves/' + track_name + '.dsv')
            i = 0
        elif pressed_key == ord('r'):
            emu.open('ROM/Mario Kart DS.nds')
            i = 0
        else:
            emu.input.keypad_add_key(key_dict.get(pressed_key, Keys.KEY_NONE))


# # https://tasvideos.org/GameResources/DS/MarioKartDS
#     emu = DeSmuME()
#     mem = emu.memory
#
#
#     emu.open('Mario Kart DS (USA) (En,Fr,De,Es,It)/Mario Kart DS (USA) (En,Fr,De,Es,It).nds')
#     emu.volume_set(0)
#     window = emu.create_sdl_window()
#     emu.input.keypad_rm_key(Keys.NO_KEY_SET)
#
#     # implement early stopping if the player hasn't crossed a checkpoint in a while
#     i = 0
#     while not window.has_quit():
#         i += 1
#         window.process_input()
#         emu.cycle()
#         window.draw()
#         time.sleep(.01)
#
#         # get the 4-byte
#         player_data_ptr = mem.signed.read_long(PLAYER_DATA_ADDR)
#         player_speed = mem.signed.read_long(player_data_ptr + PLAYER_SPEED_OFFSET)
#         checkpoint_ptr = mem.signed.read_long(CHECKPOINT_DATA_ADDR)
#         checkpoint = mem.unsigned.read_byte(checkpoint_ptr + CHECKPOINT_OFFSET)
#
#         print('Speed = ', player_speed)
#         print('Checkpoint = ', checkpoint)







