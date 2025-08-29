import os
import sys
import pygame
from editor import SimpleEditor

def main():
    app = SimpleEditor()

    # Allow drop to import
    try:
        pygame.event.set_allowed([pygame.QUIT, pygame.DROPFILE, pygame.MOUSEBUTTONDOWN,
                                  pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL,
                                  pygame.KEYDOWN, pygame.VIDEORESIZE])
    except Exception:
        pass

    # Preload from CLI
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        if sys.argv[1].endswith('.pdaw'):
            # Load project file
            app.project_path = sys.argv[1]
            app.open_project()
        else:
            # Load audio file
            from audio_utils import load_audio
            y, _ = load_audio(sys.argv[1])
            if y.size > 0:
                from models import Clip
                c = Clip(y, os.path.basename(sys.argv[1]))
                c.track_index = 0
                app.tracks[0].clips.append(c)
                app.select_clip(c)
                app.mark_project_modified()
            else:
                app.set_status(f"Failed to load audio from {sys.argv[1]}", ok=False)

    clock = pygame.time.Clock()
    while True:
        dt = clock.tick(60)
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                # Check if we need to save before quitting
                if app.project_modified:
                    # TODO: Add confirmation dialog
                    pass
                pygame.quit()
                sys.exit()
            if e.type == pygame.DROPFILE:
                if e.file.endswith('.pdaw'):
                    app.project_path = e.file
                    app.open_project()
                else:
                    mx, my = pygame.mouse.get_pos()
                    app._import_at_screen_pos(e.file, mx, my)
                continue
            app.handle_event(e)

        app.update(dt, events)
        app.draw()
        pygame.display.flip()

if __name__ == "__main__":
    main()
