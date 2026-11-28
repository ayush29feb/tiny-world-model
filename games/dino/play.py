"""Play the dino game in a window.

    python -m games.dino.play

Space or Up to jump, hold Down to duck (or to fall faster mid-jump).
After game over, press Space to play again. Esc or Q quits.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import pygame
except ImportError:
    sys.exit("Playing needs pygame-ce. Install it with:  pip install -e '.[play]'")

from games.dino import Action, DinoGame

FOREGROUND = (83, 83, 83)  # the Chrome dino's grey
BACKGROUND = (247, 247, 247)
RESTART_DELAY_FRAMES = 12  # stops a jump held at the moment of death from restarting


def to_surface(frame: np.ndarray, scale: int) -> pygame.Surface:
    """Turn a bool bitmap into a pygame surface, enlarged with hard pixel edges."""
    rgb = np.where(frame[..., None], FOREGROUND, BACKGROUND).astype(np.uint8)
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))  # pygame wants (x, y)
    height, width = frame.shape
    return pygame.transform.scale(surface, (width * scale, height * scale))


def read_action(keys) -> Action:
    if keys[pygame.K_DOWN]:
        return Action.DUCK
    if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
        return Action.JUMP
    return Action.NOOP


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the dino game.")
    parser.add_argument("--scale", type=int, default=6, help="screen pixels per game pixel")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None, help="make the first game repeatable")
    args = parser.parse_args()

    game = DinoGame(seed=args.seed)
    frame = game.reset()

    pygame.init()
    height, width = frame.shape
    screen = pygame.display.set_mode((width * args.scale, height * args.scale))
    pygame.display.set_caption("dino")
    clock = pygame.time.Clock()

    frames_since_death = 0
    running = True
    while running:
        restart = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif (
                    game.done
                    and frames_since_death >= RESTART_DELAY_FRAMES
                    and event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_RETURN)
                ):
                    restart = True

        if restart:
            frame = game.reset()
            frames_since_death = 0
        elif game.done:
            frames_since_death += 1
        else:
            frame = game.step(read_action(pygame.key.get_pressed()))

        screen.blit(to_surface(frame, args.scale), (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
