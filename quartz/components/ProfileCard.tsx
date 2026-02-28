import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import style from "./styles/profileCard.scss"

interface SocialLink {
  label: string
  href: string
  iconSrc?: string
  iconAlt?: string
}

interface Options {
  imageSrc?: string
  imageAlt?: string
  bio?: string
  links?: SocialLink[]
}

const defaultLinks: SocialLink[] = [
  { label: "in", href: "https://www.linkedin.com/in/your-id" },
  { label: "gh", href: "https://github.com/your-id" },
  { label: "mail", href: "mailto:you@example.com" },
]

export default ((userOpts?: Options) => {
  const ProfileCard: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    const imageSrc = userOpts?.imageSrc
    const imageAlt = userOpts?.imageAlt ?? "Profile image"
    const bio =
      userOpts?.bio ??
      "Write a short bio here. Add your profile image and social links in quartz.layout.ts."
    const links = userOpts?.links ?? defaultLinks

    return (
      <section class={classNames(displayClass, "profile-card")}>
        {imageSrc ? (
          <img class="profile-image" src={imageSrc} alt={imageAlt} loading="lazy" />
        ) : (
          <div class="profile-image profile-image--placeholder">{imageAlt}</div>
        )}
        <p class="profile-bio">{bio}</p>
        <ul class="profile-links">
          {links.map((link) => {
            const external = /^https?:\/\//.test(link.href)
            return (
              <li>
                <a
                  href={link.href}
                  target={external ? "_blank" : undefined}
                  rel={external ? "noopener noreferrer" : undefined}
                  aria-label={link.label}
                >
                  {link.iconSrc ? (
                    <img
                      class="profile-link-icon"
                      src={link.iconSrc}
                      alt={link.iconAlt ?? link.label}
                      loading="lazy"
                    />
                  ) : (
                    link.label
                  )}
                </a>
              </li>
            )
          })}
        </ul>
      </section>
    )
  }

  ProfileCard.css = style
  return ProfileCard
}) satisfies QuartzComponentConstructor
